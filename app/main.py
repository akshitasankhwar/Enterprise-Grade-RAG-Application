"""FastAPI Main Application - Document upload, query and health endpoints"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
from pathlib import Path
from typing import List
import shutil
import time

from app.config import settings
from app.agents.graph import create_graph, run_agent
from app.ingestion.processor import DocumentProcessor, SUPPORTED_EXTENSIONS
from app.services.qdrant_service import QdrantService
from app.models import (
    QueryRequest, QueryResponse, DocumentReference, HealthResponse, ErrorResponse,
    DocumentInfo, UploadResponse, DocumentListResponse, DeleteResponse
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Where uploaded files are saved before ingestion
UPLOAD_DIR = Path("data/uploads")

# Each answered query is appended here in the background
QUERY_LOG = Path("logs/query_log.txt")

# Global agent graph (lazy load)
agent_graph = None


def get_agent_graph():
    """Lazy load the agent graph on first use"""
    global agent_graph
    if agent_graph is None:
        logger.info("Creating agent graph...")
        agent_graph = create_graph()
    return agent_graph


# Global document processor (lazy load - holds the embedding model)
document_processor = None


def get_document_processor():
    """Lazy load the document processor on first upload"""
    global document_processor
    if document_processor is None:
        logger.info("Creating document processor...")
        document_processor = DocumentProcessor()
    return document_processor


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: prepare the collection and agent. Shutdown: log it."""
    logger.info("🚀 Starting Enterprise Agentic RAG API")
    logger.info(f"   Qdrant: {settings.QDRANT_URL or f'{settings.QDRANT_HOST}:{settings.QDRANT_PORT}'}")
    logger.info(f"   Model: {settings.EMBEDDINGS_MODEL}")

    # Create the collection up front so /health and /documents work before the first upload
    try:
        QdrantService().create_collection(vector_size=384)
    except Exception as e:
        logger.error(f"⚠️ Qdrant not reachable at startup: {e}")

    get_agent_graph()
    yield
    logger.info("🛑 Shutting down API")


# Create FastAPI app
app = FastAPI(
    title="Enterprise Agentic RAG",
    description="Production-grade RAG system with LangGraph and Groq",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (restrict in production)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint - verify all services are running.
    
    Returns:
        HealthResponse with status of all services
    """
    try:
        # Check Qdrant connection
        qdrant_service = QdrantService()
        qdrant_info = qdrant_service.get_collection_info()
        qdrant_ok = len(qdrant_info) > 0
        
        # Check settings
        settings_ok = bool(settings.GROQ_API_KEY)
        
        services = {
            "qdrant": qdrant_ok,
            "settings": settings_ok,
            "agent": agent_graph is not None
        }
        
        all_ok = all(services.values())
        status = "healthy" if all_ok else "degraded"
        
        return HealthResponse(
            status=status,
            message="All services operational" if all_ok else "Some services unavailable",
            services=services
        )
    
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="unhealthy",
            message=f"Health check failed: {str(e)}",
            services={
                "qdrant": False,
                "settings": False,
                "agent": False
            }
        )


@app.post("/query", response_model=QueryResponse, tags=["Query"])
async def query(
    request: QueryRequest,
    background_tasks: BackgroundTasks
):
    """
    Main query endpoint - Ask the agent a question.
    
    This endpoint:
    1. Validates the query
    2. Runs the agent pipeline
    3. Returns response with sources and reasoning
    
    Args:
        request: QueryRequest with query string
        background_tasks: For async logging
        
    Returns:
        QueryResponse with answer and metadata
        
    Raises:
        HTTPException: If query processing fails
    """
    try:
        logger.info(f"📥 Received query: {request.query[:100]}...")
        start_time = time.time()
        
        # Length is checked by QueryRequest; whitespace-only still gets through it
        if not request.query.strip():
            raise HTTPException(
                status_code=400,
                detail="Query cannot be empty"
            )

        # Get agent graph
        graph = get_agent_graph()
        
        # Run agent
        logger.info("Running agent pipeline...")
        result_state = run_agent(request.query, graph, request.session_id)
        
        # Convert reranked documents to response format
        sources = [
            DocumentReference(
                text=doc["text"],
                source=doc.get("source", "Unknown"),
                score=doc.get("score", 0.0),
                page=doc.get("page")
            )
            for doc in result_state["reranked_docs"]
        ]
        
        # Create response
        response = QueryResponse(
            query=request.query,
            response=result_state["response"],
            classification=result_state["classification"],
            sources=sources,
            reasoning=result_state["reasoning"],
            session_id=request.session_id
        )
        
        elapsed_time = time.time() - start_time
        logger.info(f"✅ Query completed in {elapsed_time:.2f}s")
        logger.info(f"   Classification: {result_state['classification']}")
        logger.info(f"   Sources used: {len(sources)}")
        
        # Log to background (non-blocking)
        background_tasks.add_task(
            log_query_to_file,
            request.query,
            result_state["response"],
            elapsed_time
        )
        
        return response
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error processing query: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query: {str(e)}"
        )


@app.post("/upload", response_model=UploadResponse, tags=["Documents"])
def upload_documents(files: List[UploadFile] = File(...)):
    """
    Upload documents and add them to the knowledge base.

    Each file is saved to data/uploads/, then loaded → chunked → embedded
    → stored in Qdrant. Uploading a file with the same name again replaces
    its previous chunks.

    Args:
        files: One or more files (PDF, DOCX, TXT, MD, CSV, HTML)

    Returns:
        UploadResponse with chunk counts per file

    Raises:
        HTTPException: If no supported files were sent or ingestion fails
    """
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    saved_paths = []
    skipped = []

    for file in files:
        # Keep only the base name so a crafted filename can't escape UPLOAD_DIR
        filename = Path(file.filename or "").name

        if not filename or Path(filename).suffix.lower() not in SUPPORTED_EXTENSIONS:
            skipped.append(file.filename or "unnamed")
            continue

        file_path = UPLOAD_DIR / filename
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        saved_paths.append(str(file_path))

    if not saved_paths:
        raise HTTPException(
            status_code=400,
            detail=f"No supported files. Supported types: {', '.join(SUPPORTED_EXTENSIONS)}"
        )

    try:
        logger.info(f"📤 Ingesting {len(saved_paths)} uploaded file(s)...")
        chunk_counts = get_document_processor().process_files(saved_paths)
    except Exception as e:
        logger.error(f"❌ Error ingesting uploads: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error ingesting files: {str(e)}"
        )

    # Files with no extractable text aren't in the knowledge base, so don't keep them
    for filename, count in chunk_counts.items():
        if not count:
            (UPLOAD_DIR / filename).unlink(missing_ok=True)

    uploaded = [
        DocumentInfo(filename=filename, chunks=count)
        for filename, count in chunk_counts.items()
    ]
    logger.info(f"✅ Ingested {sum(chunk_counts.values())} chunks from {len(uploaded)} file(s)")

    return UploadResponse(
        files=uploaded,
        skipped=skipped,
        total_chunks=sum(chunk_counts.values())
    )


@app.get("/documents", response_model=DocumentListResponse, tags=["Documents"])
def list_documents():
    """
    List the documents in the knowledge base.

    Returns:
        DocumentListResponse with each file's chunk count
    """
    try:
        counts = QdrantService().list_sources()
    except Exception as e:
        logger.error(f"Error listing documents: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error listing documents: {str(e)}"
        )

    return DocumentListResponse(
        documents=[
            DocumentInfo(filename=filename, chunks=count)
            for filename, count in sorted(counts.items(), key=lambda item: item[0].lower())
        ],
        total_chunks=sum(counts.values())
    )


@app.delete("/documents/{filename}", response_model=DeleteResponse, tags=["Documents"])
def delete_document(filename: str):
    """
    Remove a document from the knowledge base and delete its stored file.

    Args:
        filename: Name of the uploaded file

    Returns:
        DeleteResponse with the number of chunks removed

    Raises:
        HTTPException: If the document isn't in the knowledge base
    """
    filename = Path(filename).name
    qdrant_service = QdrantService()

    chunk_count = qdrant_service.list_sources().get(filename, 0)
    if not chunk_count:
        raise HTTPException(
            status_code=404,
            detail=f"Document not found: {filename}"
        )

    qdrant_service.delete_by_source(filename)
    (UPLOAD_DIR / filename).unlink(missing_ok=True)
    logger.info(f"🗑️ Deleted {filename} ({chunk_count} chunks)")

    return DeleteResponse(filename=filename, deleted_chunks=chunk_count)


@app.get("/info", tags=["Info"])
async def info():
    """
    Get API and configuration information.
    
    Returns:
        API metadata and configuration
    """
    try:
        qdrant_service = QdrantService()
        collection_info = qdrant_service.get_collection_info()
        
        return {
            "api_name": "Enterprise Agentic RAG",
            "version": "1.0.0",
            "qdrant": {
                "url": settings.QDRANT_URL or f"http://{settings.QDRANT_HOST}:{settings.QDRANT_PORT}",
                "collection": settings.QDRANT_COLLECTION,
                "point_count": collection_info.get("point_count", 0),
                "vector_size": collection_info.get("vector_size", 0)
            },
            "embeddings_model": settings.EMBEDDINGS_MODEL,
            "status": "ready"
        }
    except Exception as e:
        logger.error(f"Error getting info: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error getting info: {str(e)}"
        )


@app.get("/", tags=["Root"])
async def root():
    """Welcome endpoint"""
    return {
        "message": "Welcome to Enterprise Agentic RAG",
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "upload": "POST /upload",
            "documents": "GET /documents",
            "delete_document": "DELETE /documents/{filename}",
            "query": "POST /query",
            "health": "GET /health",
            "info": "GET /info"
        }
    }


# Background tasks
def log_query_to_file(query: str, response: str, elapsed_time: float):
    """Log query and response to file (background task)"""
    try:
        QUERY_LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(QUERY_LOG, "a", encoding="utf-8") as f:
            f.write(f"\n{'='*60}\n")
            f.write(f"Query: {query}\n")
            f.write(f"Response: {response[:200]}...\n")
            f.write(f"Time: {elapsed_time:.2f}s\n")
    except Exception as e:
        logger.error(f"Error logging to file: {e}")


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions"""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.detail,
            status_code=exc.status_code
        ).model_dump()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            detail=str(exc),
            status_code=500
        ).model_dump()
    )
