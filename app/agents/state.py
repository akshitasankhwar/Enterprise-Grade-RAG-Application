"""Agent State - Define what flows through the LangGraph"""

from dataclasses import dataclass, field
from typing import List, Dict, Any

from app.config import settings


@dataclass
class AgentState:
    """
    State object passed through LangGraph nodes.
    This holds all the data as it flows through the agent.
    """

    # Input
    query: str
    """The user's question"""

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
    model_name: str = settings.GROQ_MODEL
    """Which Groq model to use (GROQ_MODEL in .env)"""
