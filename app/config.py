"""Configuration - Load environment variables"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional


class Settings(BaseSettings):
    """Application settings from .env file"""
    
    # Project
    PROJECT_ID: str = "local-rag"
    LOCATION: str = "local"
    
    # LLM (Groq)
    GROQ_API_KEY: str  # Required!
    
    # Embeddings
    EMBEDDINGS_MODEL: str = "all-MiniLM-L6-v2"
    
    # Qdrant (Vector DB)
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_COLLECTION: str = "documents"

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


if __name__ == "__main__":
    print(f"Project: {settings.PROJECT_ID}")
    print(f"Qdrant: {settings.QDRANT_HOST}:{settings.QDRANT_PORT}")
    print(f"Collection: {settings.QDRANT_COLLECTION}")
    print(f"Embeddings model: {settings.EMBEDDINGS_MODEL}")