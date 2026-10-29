"""LLM - Groq chat model shared by the agent nodes"""

from langchain_groq import ChatGroq

from app.config import settings


def make_llm(model_name: str, temperature: float) -> ChatGroq:
    """Create the Groq chat model"""
    return ChatGroq(
        model_name=model_name,
        temperature=temperature,
        groq_api_key=settings.GROQ_API_KEY
    )
