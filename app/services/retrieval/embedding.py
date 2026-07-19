"""Embedding Service - Create embeddings using local sentence-transformers"""

from sentence_transformers import SentenceTransformer
from typing import List
import os


class EmbeddingService:
    """
    Create embeddings using sentence-transformers (local, no API key needed).
    """
    
    def __init__(self, model_name: str = None):
        """
        Initialize embedding service.
        
        Args:
            model_name: Name of the sentence-transformer model
                       Defaults to all-MiniLM-L6-v2 (fast, small)
        """
        if model_name is None:
            model_name = os.getenv("EMBEDDINGS_MODEL", "all-MiniLM-L6-v2")
        
        print(f"Loading embedding model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        print(f"✅ Model loaded. Embedding dimension: {self.model.get_sentence_embedding_dimension()}")
    
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


if __name__ == "__main__":
    # Test the embedding service
    service = EmbeddingService()
    
    # Test single text
    text = "This is a test document about machine learning."
    embedding = service.embed_text(text)
    print(f"Single embedding shape: {len(embedding)}")
    
    # Test multiple texts
    texts = [
        "Machine learning is about data",
        "Deep learning uses neural networks",
        "NLP processes natural language"
    ]
    embeddings = service.embed_texts(texts)
    print(f"Multiple embeddings shape: {len(embeddings)} x {len(embeddings[0])}")