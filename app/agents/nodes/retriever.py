"""Retriever Node - Fetch relevant documents from Qdrant"""

from app.agents.state import AgentState
from app.services.retrieval.embedding import EmbeddingService
from app.services.retrieval.qdrant_service import QdrantService


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
    print(f"Query: {state.query}")
    
    try:
        # Initialize services
        embedding_service = EmbeddingService()
        qdrant_service = QdrantService()
        
        # Embed the query
        print("Creating embedding for query...")
        query_vector = embedding_service.embed_text(state.query)
        
        # Search Qdrant
        print("Searching Qdrant...")
        results = qdrant_service.search(query_vector, limit=10)
        
        if results:
            state.retrieved_docs = results
            print(f"✅ Found {len(results)} documents")
            
            # Show top result
            top_result = results[0]
            print(f"   Top match score: {top_result['score']:.4f}")
            print(f"   Source: {top_result['source']}")
            print(f"   Text: {top_result['text'][:100]}...")
        else:
            print("⚠️ No documents found in database")
            state.retrieved_docs = []
        
        state.reasoning += f" → Retrieved {len(results)} documents"
        
    except Exception as e:
        print(f"❌ Error in retriever: {e}")
        state.retrieved_docs = []
        state.reasoning += f" → Retrieval error: {str(e)}"
    
    return state


if __name__ == "__main__":
    # Test the retriever
    from app.config import settings
    
    state = AgentState(
        query="What is machine learning?",
        classification="TECHNICAL"
    )
    
    result = retriever_node(state)
    print(f"\nRetrieved {len(result.retrieved_docs)} documents")
    
    if result.retrieved_docs:
        for i, doc in enumerate(result.retrieved_docs[:3], 1):
            print(f"\n{i}. Score: {doc['score']:.4f}")
            print(f"   Source: {doc['source']}")
            print(f"   Text: {doc['text'][:100]}...")