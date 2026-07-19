"""FastAPI Main Application - Query and health endpoints"""

from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
from typing import Optional
import time

from app.config import settings
from app.agents.graph import create_graph, run_agent
from app.agents.state import AgentState
from app.services.retrieval.qdrant_service import QdrantService
from app.models import QueryRequest, QueryResponse, DocumentReference, HealthResponse, ErrorResponse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Enterprise Agentic RAG",
    description="Production-grade RAG system with LangGraph and Groq",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (restrict in production)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global agent graph (lazy load)
agent_graph = None


def get_agent_graph():
    """Lazy load the agent graph on first use"""
    global agent_graph
    if agent_graph is None:
        logger.info("Creating agent graph...")
        agent_graph = create_graph()
    return agent_graph


@app.on_event("startup")
async def startup_event():
    """Run on startup - initialize services"""
    logger.info("🚀 Starting Enterprise Agentic RAG API")
    logger.info(f"   Project: {settings.PROJECT_ID}")
    logger.info(f"   Qdrant: {settings.QDRANT_HOST}:{settings.QDRANT_PORT}")
    logger.info(f"   Model: {settings.EMBEDDINGS_MODEL}")


@app.on_event("shutdown")
async def shutdown_event():
    """Run on shutdown"""
    logger.info("🛑 Shutting down API")


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
        
        # Validate query
        if not request.query.strip():
            raise HTTPException(
                status_code=400,
                detail="Query cannot be empty"
            )
        
        if len(request.query) > 2000:
            raise HTTPException(
                status_code=400,
                detail="Query too long (max 2000 characters)"
            )
        
        # Get agent graph
        graph = get_agent_graph()
        
        # Run agent
        logger.info("Running agent pipeline...")
        result_state = run_agent(request.query, graph)
        
        # Convert reranked documents to response format
        sources = [
            DocumentReference(
                text=doc["text"],
                source=doc.get("source", "Unknown"),
                score=doc.get("score", 0.0)
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
            "project": settings.PROJECT_ID,
            "location": settings.LOCATION,
            "qdrant": {
                "host": settings.QDRANT_HOST,
                "port": settings.QDRANT_PORT,
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


@app.get("/collections", tags=["Info"])
async def list_collections():
    """
    List all Qdrant collections.
    
    Returns:
        List of collections with metadata
    """
    try:
        qdrant_service = QdrantService()
        info = qdrant_service.get_collection_info()
        
        return {
            "collections": [
                {
                    "name": info.get("name"),
                    "points": info.get("point_count"),
                    "vector_size": info.get("vector_size")
                }
            ]
        }
    except Exception as e:
        logger.error(f"Error listing collections: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error listing collections: {str(e)}"
        )


@app.get("/", tags=["Root"])
async def root():
    """Welcome endpoint"""
    return {
        "message": "Welcome to Enterprise Agentic RAG",
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "query": "POST /query",
            "health": "GET /health",
            "info": "GET /info",
            "collections": "GET /collections"
        }
    }


# Background tasks
def log_query_to_file(query: str, response: str, elapsed_time: float):
    """Log query and response to file (background task)"""
    try:
        with open("query_log.txt", "a") as f:
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
        ).dict()
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
        ).dict()
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        reload=True  # Enable auto-reload during development
    )