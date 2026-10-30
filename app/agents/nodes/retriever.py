"""Retriever Node - Fetch relevant documents from Qdrant"""

from app.config import settings
from app.agents.state import AgentState
from app.services.embedding_service import get_embedding_service
from app.services.qdrant_service import QdrantService


def retriever_node(state: AgentState) -> AgentState:
    """
    Retriever Node: Search Qdrant for documents relevant to the query.
    
    Steps:
    1. Embed the query
    2. Search Qdrant for similar documents
    3. Store results in state
    
    Args:
        state: Current agent state
        
    Returns:
        Updated state with retrieved documents
    """
    print("\n🔍 RETRIEVER NODE")
    print(f"Query: {state.search_query}")
    
    try:
        # Initialize services
        embedding_service = get_embedding_service()
        qdrant_service = QdrantService()
        
        # Embed the query
        print("Creating embedding for query...")
        query_vector = embedding_service.embed_text(state.search_query)
        
        # Search Qdrant, skipping passages below the relevance floor
        print("Searching Qdrant...")
        results = qdrant_service.search(query_vector, limit=10, score_threshold=settings.MIN_RELEVANCE)

        if results:
            state.retrieved_docs = results
            print(f"✅ Found {len(results)} documents")

            # Show top result
            top_result = results[0]
            print(f"   Top match score: {top_result['score']:.4f}")
            print(f"   Source: {top_result['source']}")
            print(f"   Text: {top_result['text'][:100]}...")
        else:
            print(f"⚠️ No passages with similarity ≥ {settings.MIN_RELEVANCE}")
            state.retrieved_docs = []

        state.reasoning += f" → Retrieved {len(results)} passages (similarity ≥ {settings.MIN_RELEVANCE})"
        
    except Exception as e:
        print(f"❌ Error in retriever: {e}")
        state.retrieved_docs = []
        state.reasoning += f" → Retrieval error: {str(e)}"
    
    return state
