"""Responder Node - Generate the final answer"""
from app.config import settings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from app.agents.state import AgentState
from app.services.retrieval.ranking_service import RankingService


def responder_node(state: AgentState) -> AgentState:
    """
    Responder Node: Generate the final answer based on:
    - Retrieved documents (if technical)
    - Query and conversation history
    - Classification (conversational vs technical)
    
    Steps:
    1. Rerank retrieved documents (if any)
    2. Create context from top documents
    3. Generate response using LLM
    4. Add to conversation history
    
    Args:
        state: Current agent state
        
    Returns:
        Updated state with response
    """
    print("\n💬 RESPONDER NODE")
    print(f"Classification: {state.classification}")
    
    try:
        # Initialize LLM
        llm = ChatGroq(
            model_name=state.model_name,
            temperature=0.7,  # Some creativity for responses
            groq_api_key=settings.GROQ_API_KEY  # Uses GROQ_API_KEY from environment
        )
        
        # If technical query and we have documents, rerank them
        context = ""
        if state.classification == "TECHNICAL" and state.retrieved_docs:
            print("Reranking documents...")
            ranking_service = RankingService()
            
            doc_texts = [doc["text"] for doc in state.retrieved_docs]
            reranked = ranking_service.rerank(state.query, doc_texts, top_k=3)
            
            state.reranked_docs = [
                {
                    "text": r["text"],
                    "score": r["score"],
                    "source": state.retrieved_docs[r["index"]].get("source", "Unknown")
                }
                for r in reranked
            ]
            
            print(f"✅ Reranked to {len(state.reranked_docs)} documents")
            
            # Build context from reranked documents
            context_parts = []
            for doc in state.reranked_docs:
                context_parts.append(
                    f"Source: {doc['source']}\n{doc['text']}"
                )
            context = "\n\n---\n\n".join(context_parts)
        
        # Create response prompt
        if state.classification == "TECHNICAL" and context:
            # Technical with documents
            prompt = ChatPromptTemplate.from_template("""
You are a helpful AI assistant. Answer the user's question based on the provided documents.

DOCUMENTS:
{context}

QUESTION: {query}

ANSWER: Provide a clear, concise answer based on the documents. If the documents don't contain the answer, say so honestly.
""")
            
            chain = prompt | llm
            response = chain.invoke({
                "query": state.query,
                "context": context
            })
        
        elif state.classification == "CONVERSATIONAL":
            # Conversational - no documents needed
            prompt = ChatPromptTemplate.from_template("""
You are a friendly and helpful AI assistant.

Question: {query}

Answer:
""")
            
            chain = prompt | llm
            response = chain.invoke({"query": state.query})
        
        else:
            # Technical but no documents found
            prompt = ChatPromptTemplate.from_template("""
You are a helpful AI assistant.

The user asked: {query}

Unfortunately, I don't have relevant documents to answer this. However, I can provide general knowledge:

Answer:
""")
            
            chain = prompt | llm
            response = chain.invoke({"query": state.query})
        
        state.response = response.content
        
        # Update history
        state.add_to_history("user", state.query)
        state.add_to_history("assistant", state.response)
        
        # Update reasoning
        if state.reranked_docs:
            state.reasoning += f" → Reranked {len(state.reranked_docs)} docs → Generated response"
        else:
            state.reasoning += f" → Generated response"
        
        print(f"✅ Generated response")
        print(f"   Length: {len(state.response)} characters")
        
    except Exception as e:
        print(f"❌ Error in responder: {e}")
        state.response = f"I encountered an error: {str(e)}"
        state.reasoning += f" → Error: {str(e)}"
    
    return state


if __name__ == "__main__":
    # Test the responder
    from app.config import settings
    
    # Test conversational
    print("=" * 60)
    print("TEST 1: CONVERSATIONAL")
    print("=" * 60)
    state1 = AgentState(
        query="Hello! How are you?",
        classification="CONVERSATIONAL"
    )
    result1 = responder_node(state1)
    print(f"\nResponse: {result1.response}\n")
    
    # Test technical (with mock documents)
    print("=" * 60)
    print("TEST 2: TECHNICAL")
    print("=" * 60)
    state2 = AgentState(
        query="What is machine learning?",
        classification="TECHNICAL",
        retrieved_docs=[
            {
                "text": "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience.",
                "source": "ml_basics.txt",
                "score": 0.95
            }
        ]
    )
    result2 = responder_node(state2)
    print(f"\nResponse: {result2.response}\n")