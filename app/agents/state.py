"""Agent State - Define what flows through the LangGraph"""

from dataclasses import dataclass, field
from typing import List, Dict, Any


@dataclass
class AgentState:
    """
    State object passed through LangGraph nodes.
    This holds all the data as it flows through the agent.
    """
    
    # Input
    query: str
    """The user's question"""
    
    # History
    history: List[Dict[str, str]] = field(default_factory=list)
    """Conversation history: [{"role": "user", "content": "..."}, ...]"""
    
    # Intermediate results
    classification: str = ""
    """CONVERSATIONAL or TECHNICAL (decided by Planner)"""
    
    retrieved_docs: List[Dict[str, Any]] = field(default_factory=list)
    """Raw search results from Qdrant"""
    
    reranked_docs: List[Dict[str, Any]] = field(default_factory=list)
    """Reranked documents from FlashRank"""
    
    # Output
    response: str = ""
    """The final answer to the user"""
    
    reasoning: str = ""
    """Explanation of what the agent did"""
    
    # Metadata
    model_name: str = "llama-3.3-70b-versatile"
    """Which Groq model to use"""
    
    def add_to_history(self, role: str, content: str):
        """Add a message to conversation history."""
        self.history.append({"role": role, "content": content})
    
    def get_context(self) -> str:
        """Get formatted context from reranked documents."""
        if not self.reranked_docs:
            return ""
        
        context = "\n\n".join([
            f"Source: {doc.get('source', 'Unknown')}\n{doc.get('text', '')}"
            for doc in self.reranked_docs
        ])
        return context


if __name__ == "__main__":
    # Test the state
    state = AgentState(query="What is machine learning?")
    state.add_to_history("user", "What is machine learning?")
    state.classification = "TECHNICAL"
    state.retrieved_docs = [{"text": "ML is...", "source": "doc1.txt"}]
    state.response = "Machine learning is..."
    
    print(f"Query: {state.query}")
    print(f"Classification: {state.classification}")
    print(f"Response: {state.response}")
    print(f"History: {state.history}")