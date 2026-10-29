"""Qdrant Vector Database Service - Store and search embeddings"""

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, PayloadSchemaType,
    Filter, FieldCondition, MatchValue, FilterSelector
)
from typing import List, Dict, Optional
import uuid

from app.config import settings


# Shared client (lazy load) - one connection pool per process
_client = None


def get_client() -> QdrantClient:
    """
    Get the process-wide Qdrant client.

    Uses the Qdrant Cloud cluster at QDRANT_URL (with QDRANT_API_KEY) if set,
    otherwise the server at QDRANT_HOST:QDRANT_PORT.
    """
    global _client
    if _client is None:
        try:
            if settings.QDRANT_URL:
                print(f"Connecting to Qdrant at {settings.QDRANT_URL}...")
                _client = QdrantClient(url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY)
            else:
                print(f"Connecting to Qdrant at {settings.QDRANT_HOST}:{settings.QDRANT_PORT}...")
                _client = QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)
            print("✅ Connected to Qdrant")
        except Exception as e:
            print(f"❌ Failed to connect to Qdrant: {e}")
            raise
    return _client


class QdrantService:
    """
    Manage vector storage and retrieval with Qdrant.
    """

    def __init__(self):
        """Use the shared client and the configured collection."""
        self.client = get_client()
        self.collection_name = settings.QDRANT_COLLECTION
    
    def create_collection(self, vector_size: int = 384):
        """
        Create the collection for storing embeddings if it doesn't exist yet.

        Args:
            vector_size: Dimension of embeddings (384 for all-MiniLM-L6-v2)
        """
        try:
            if self.client.collection_exists(self.collection_name):
                print(f"✅ Collection '{self.collection_name}' already exists")
            else:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
                )
                print(f"✅ Created collection '{self.collection_name}' with vector size {vector_size}")

            # Listing and deleting filter on the file name; Qdrant Cloud's strict mode can
            # reject filters on unindexed fields. Creating an existing index is a no-op.
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="source",
                field_schema=PayloadSchemaType.KEYWORD,
                wait=True
            )

        except Exception as e:
            print(f"❌ Error creating collection: {e}")
            raise
    
    def add_documents(self, documents: List[Dict], embeddings: List[List[float]]):
        """
        Add documents with their embeddings to Qdrant in batches.
        """

        try:
            points = []

            for i, (doc, emb) in enumerate(zip(documents, embeddings)):
                point = PointStruct(
                    id=int(uuid.uuid4().int % (2**63)),
                    vector=emb,
                    payload={
                        "text": doc.get("text", ""),
                        "source": doc.get("source", "unknown"),
                        "page": doc.get("page", 0),
                        "chunk_index": doc.get("chunk_index", i),
                    }
                    )
                points.append(point)

            total = len(points)
            batch_size = 100

            print(f"\nUploading {total} vectors in batches of {batch_size}...")

            for start in range(0, total, batch_size):

                batch = points[start:start + batch_size]

                self.client.upsert(
                    collection_name=self.collection_name,
                    points=batch,
                    wait=True
                )

                print(f"✅ Uploaded {min(start+batch_size,total)}/{total}")

            print("🎉 All documents uploaded successfully!")

        except Exception as e:
            print(f"❌ Error adding documents: {repr(e)}")
            raise
    
    def delete_by_source(self, source: str):
        """
        Delete all chunks that came from a given source file.

        Args:
            source: The source file name stored in the chunk payload
        """
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=FilterSelector(
                    filter=Filter(
                        must=[FieldCondition(key="source", match=MatchValue(value=source))]
                    )
                ),
                wait=True
            )
        except Exception as e:
            print(f"❌ Error deleting chunks for {source}: {e}")
            raise

    def list_sources(self) -> Dict[str, int]:
        """
        Count stored chunks per source file.

        Returns:
            Mapping of source file name → chunk count (empty if no collection yet)
        """
        if not self.client.collection_exists(self.collection_name):
            return {}

        counts = {}
        offset = None
        while True:
            points, offset = self.client.scroll(
                collection_name=self.collection_name,
                limit=1000,
                offset=offset,
                with_payload=["source"],
                with_vectors=False
            )
            for point in points:
                source = point.payload.get("source", "unknown")
                counts[source] = counts.get(source, 0) + 1
            if offset is None:
                break
        return counts

    def search(
        self,
        query_vector: List[float],
        limit: int = 5,
        score_threshold: Optional[float] = None
    ) -> List[Dict]:
        """
        Search for similar documents.

        Args:
            query_vector: Query embedding
            limit: Number of results to return
            score_threshold: Minimum cosine similarity; less similar passages are dropped

        Returns:
            List of similar documents with scores
        """
        try:
            results = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=limit,
                score_threshold=score_threshold
            )
            
            return [
                {
                    "text": result.payload.get("text", ""),
                    "source": result.payload.get("source", ""),
                    "page": result.payload.get("page"),
                    "score": result.score,
                    "id": result.id
                }
                for result in results.points
            ]
        
        except Exception as e:
            print(f"❌ Error searching: {e}")
            return []
    
    def get_collection_info(self) -> Dict:
        """Get information about the collection."""
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                "name": self.collection_name,
                "point_count": info.points_count,
                "vector_size": info.config.params.vectors.size
            }
        except Exception as e:
            print(f"❌ Error getting collection info: {e}")
            return {}
