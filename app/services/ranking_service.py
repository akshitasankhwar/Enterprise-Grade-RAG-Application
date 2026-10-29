"""Ranking Service - Rerank documents using FlashRank"""

from flashrank import Ranker, RerankRequest
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
            
            # Rerank (results come back sorted by score, highest first)
            results = self.ranker.rerank(
                RerankRequest(query=query, passages=passages)
            )

            # Return top-k
            top_results = results[:top_k] if len(results) >= top_k else results

            return [
                {
                    "text": result["text"],
                    "score": float(result["score"]),  # numpy float32 → JSON-safe float
                    "index": result["id"]
                }
                for result in top_results
            ]
        
        except Exception as e:
            print(f"❌ Error reranking: {e}")
            return []


# Shared instance (lazy load) - avoids reloading the reranker model on every query
_ranking_service = None


def get_ranking_service() -> RankingService:
    """Get the process-wide ranking service."""
    global _ranking_service
    if _ranking_service is None:
        _ranking_service = RankingService()
    return _ranking_service
