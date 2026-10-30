"""API Models - Request and Response schemas"""

from pydantic import BaseModel, ConfigDict, Field
from typing import List, Dict, Optional


class QueryRequest(BaseModel):
    """Request model for /query endpoint"""
    
    query: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The user's question or message"
    )
    
    session_id: Optional[str] = Field(
        None,
        description="Chat session ID. Queries with the same ID share memory, so follow-ups work; "
                    "omit it to ask a one-off question. Echoed back in the response"
    )

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "query": "What are the payment terms?",
            "session_id": "user_123"
        }
    })


class DocumentReference(BaseModel):
    """A source document reference"""
    
    text: str = Field(..., description="The document text chunk")
    source: str = Field(..., description="Source file name")
    score: float = Field(..., description="Relevance score (0-1)")
    page: Optional[int] = Field(None, description="Page (PDF) or section index within the file")


class QueryResponse(BaseModel):
    """Response model for /query endpoint"""
    
    query: str = Field(..., description="The original query")
    response: str = Field(..., description="The AI-generated response")
    classification: str = Field(..., description="CONVERSATIONAL or TECHNICAL")
    sources: List[DocumentReference] = Field(
        default_factory=list,
        description="Source documents used (if any)"
    )
    reasoning: str = Field(..., description="Explanation of what the agent did")
    session_id: Optional[str] = Field(None, description="Session ID for tracking")

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "query": "What are the payment terms?",
            "response": "Invoices are payable within 45 days of the invoice date [1].",
            "classification": "TECHNICAL",
            "sources": [
                {
                    "text": "Invoices are payable within 45 days of the invoice date...",
                    "source": "vendor_contract.pdf",
                    "score": 0.95,
                    "page": 0
                }
            ],
            "reasoning": "Classified as TECHNICAL → Retrieved 6 documents → Reranked 3 docs → Generated cited response"
        }
    })


class HealthResponse(BaseModel):
    """Health check response"""
    
    status: str = Field(..., description="Service status")
    message: str = Field(..., description="Status message")
    services: Dict[str, bool] = Field(..., description="Status of each service")


class DocumentInfo(BaseModel):
    """A document in the knowledge base"""

    filename: str = Field(..., description="Name of the uploaded file")
    chunks: int = Field(..., description="Number of chunks stored in Qdrant")


class UploadResponse(BaseModel):
    """Response model for /upload endpoint"""

    files: List[DocumentInfo] = Field(default_factory=list, description="Files that were ingested")
    skipped: List[str] = Field(default_factory=list, description="Files skipped (unsupported type)")
    total_chunks: int = Field(..., description="Total chunks added to the knowledge base")


class DocumentListResponse(BaseModel):
    """Response model for GET /documents"""

    documents: List[DocumentInfo] = Field(default_factory=list, description="Uploaded documents")
    total_chunks: int = Field(..., description="Total chunks in the knowledge base")


class DeleteResponse(BaseModel):
    """Response model for DELETE /documents/{filename}"""

    filename: str = Field(..., description="Name of the deleted file")
    deleted_chunks: int = Field(..., description="Number of chunks removed")


class ErrorResponse(BaseModel):
    """Error response"""
    
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
    status_code: int = Field(..., description="HTTP status code")
