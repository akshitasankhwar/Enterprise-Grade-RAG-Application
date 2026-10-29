"""Embedding Service - Create embeddings using local sentence-transformers"""

from sentence_transformers import SentenceTransformer
from typing import List

from app.config import settings


class EmbeddingService:
    """
    Create embeddings using sentence-transformers (local, no API key needed).
    """

    def __init__(self, model_name: str = None):
        """
        Initialize embedding service.

        Args:
            model_name: Name of the sentence-transformer model
                       Defaults to EMBEDDINGS_MODEL (all-MiniLM-L6-v2: fast, small)
        """
        if model_name is None:
            model_name = settings.EMBEDDINGS_MODEL

        print(f"Loading embedding model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        print(f"✅ Model loaded. Embedding dimension: {self.model.get_embedding_dimension()}")

    def embed_text(self, text: str) -> List[float]:
        """
        Embed a single text string.

        Args:
            text: Text to embed

        Returns:
            Embedding vector as list of floats
        """
        embedding = self.model.encode(text, convert_to_tensor=False)
        return embedding.tolist()

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Embed multiple texts efficiently.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        embeddings = self.model.encode(texts, convert_to_tensor=False)
        return [emb.tolist() for emb in embeddings]


# Shared instance (lazy load) - loading the model takes seconds, so do it once per process
_embedding_service = None


def get_embedding_service() -> EmbeddingService:
    """Get the process-wide embedding service."""
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service
