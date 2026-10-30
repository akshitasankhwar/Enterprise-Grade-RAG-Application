"""Configuration - Load environment variables"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional


class Settings(BaseSettings):
    """Application settings from .env file"""

    # LLM (Groq)
    GROQ_API_KEY: str  # Required!
    GROQ_MODEL: str = "openai/gpt-oss-120b"  # See https://console.groq.com/docs/models

    # Embeddings
    EMBEDDINGS_MODEL: str = "all-MiniLM-L6-v2"

    # Qdrant (Vector DB)
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_COLLECTION: str = "documents"
    QDRANT_URL: Optional[str] = None  # Qdrant Cloud cluster URL - overrides host/port when set
    QDRANT_API_KEY: Optional[str] = None  # Qdrant Cloud API key

    # Retrieval - passages less similar than this (cosine, 0-1) are never shown to the LLM
    MIN_RELEVANCE: float = 0.15

    # Memory - how many recent question/answer pairs each chat remembers (0 = no memory)
    MEMORY_TURNS: int = 5

    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=True,
        extra="ignore"
    )


# Create global settings instance
try:
    settings = Settings()
    print("✅ Settings loaded from .env")
except Exception as e:
    print(f"❌ Error loading settings: {e}")
    print("Make sure .env file exists with GROQ_API_KEY")
    raise
