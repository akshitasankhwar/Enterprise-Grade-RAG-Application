"""API Models - Request and Response schemas"""

from pydantic import BaseModel, Field
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
        description="Optional session ID for conversation history"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "What is machine learning?",
                "session_id": "user_123"
            }
        }


class DocumentReference(BaseModel):
    """A source document reference"""
    
    text: str = Field(..., description="The document text chunk")
    source: str = Field(..., description="Source file or URL")
    score: float = Field(..., description="Relevance score (0-1)")


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
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "What is machine learning?",
                "response": "Machine learning is a subset of artificial intelligence...",
                "classification": "TECHNICAL",
                "sources": [
                    {
                        "text": "ML is a subset of AI that learns from data...",
                        "source": "ml_basics.txt",
                        "score": 0.95
                    }
                ],
                "reasoning": "Classified as TECHNICAL → Retrieved 10 documents → Reranked 3 docs → Generated response"
            }
        }


class HealthResponse(BaseModel):
    """Health check response"""
    
    status: str = Field(..., description="Service status")
    message: str = Field(..., description="Status message")
    services: Dict[str, bool] = Field(..., description="Status of each service")


class ErrorResponse(BaseModel):
    """Error response"""
    
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
    status_code: int = Field(..., description="HTTP status code")


if __name__ == "__main__":
    # Test models
    query = QueryRequest(query="Hello!")
    print(f"Query: {query.query}")
    
    response = QueryResponse(
        query="What is ML?",
        response="Machine learning is...",
        classification="TECHNICAL",
        reasoning="Processed successfully"
    )
    print(f"Response: {response.response}")
