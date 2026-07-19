"""Ranking Service - Rerank documents using FlashRank"""

from flashrank import Ranker
from typing import List, Dict


class RankingService:
    """
    Rerank documents by relevance using FlashRank (super fast!).
    """
    
    def __init__(self):
        """Initialize the ranker."""
        print("Loading FlashRank...")
        try:
            self.ranker = Ranker()
            print("✅ FlashRank loaded")
        except Exception as e:
            print(f"❌ Error loading FlashRank: {e}")
            raise
    
    def rerank(self, query: str, documents: List[str], top_k: int = 3) -> List[Dict]:
        """
        Rerank documents by relevance to the query.
        
        Args:
            query: The query string
            documents: List of documents to rerank
            top_k: Number of top results to return
            
        Returns:
            List of reranked documents with scores
        """
        try:
            if not documents:
                return []
            
            # Format for flashrank
            passages = [{"id": i, "text": doc} for i, doc in enumerate(documents)]
            
            # Rerank
            results = self.ranker.rank(
                query=query,
                passages=passages,
                batch_size=min(len(passages), 128)  # Batch size for efficiency
            )
            
            # Return top-k
            top_results = results[:top_k] if len(results) >= top_k else results
            
            return [
                {
                    "text": result["text"],
                    "score": result["score"],
                    "index": result["id"]
                }
                for result in top_results
            ]
        
        except Exception as e:
            print(f"❌ Error reranking: {e}")
            return []


if __name__ == "__main__":
    # Test the service
    service = RankingService()
    
    query = "What is machine learning?"
    documents = [
        "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience.",
        "The capital of France is Paris.",
        "Deep learning uses neural networks with multiple layers.",
        "Python is a programming language."
    ]
    
    results = service.rerank(query, documents, top_k=2)
    print(f"Top {len(results)} results:")
    for i, result in enumerate(results, 1):
        print(f"{i}. Score: {result['score']:.4f} - {result['text'][:80]}...")