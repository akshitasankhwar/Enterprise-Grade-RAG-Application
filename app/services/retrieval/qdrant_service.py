"""Qdrant Vector Database Service - Store and search embeddings"""

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from typing import List, Dict
import os
import uuid


class QdrantService:
    """
    Manage vector storage and retrieval with Qdrant (LOCAL).
    """
    
    def __init__(self):
        """Initialize Qdrant client for local storage."""
        host = os.getenv("QDRANT_HOST", "localhost")
        port = int(os.getenv("QDRANT_PORT", 6333))
        
        print(f"Connecting to Qdrant at {host}:{port}...")
        try:
            self.client = QdrantClient(host=host, port=port)
            print("✅ Connected to Qdrant")
        except Exception as e:
            print(f"❌ Failed to connect to Qdrant: {e}")
            raise
        
        self.collection_name = os.getenv("QDRANT_COLLECTION", "documents")
    
    def create_collection(self, vector_size: int = 384, force_recreate: bool = False):
        """
        Create a collection for storing embeddings.
        
        Args:
            vector_size: Dimension of embeddings (384 for all-MiniLM-L6-v2)
            force_recreate: If True, delete and recreate the collection
        """
        try:
            # Check if collection exists
            collections = self.client.get_collections().collections
            collection_exists = any(c.name == self.collection_name for c in collections)
            
            if collection_exists:
                if force_recreate:
                    print(f"Deleting existing collection: {self.collection_name}")
                    self.client.delete_collection(self.collection_name)
                else:
                    print(f"✅ Collection '{self.collection_name}' already exists")
                    return
            
            # Create collection
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )
            print(f"✅ Created collection '{self.collection_name}' with vector size {vector_size}")
        
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
    
    def search(self, query_vector: List[float], limit: int = 5) -> List[Dict]:
        """
        Search for similar documents.
        
        Args:
            query_vector: Query embedding
            limit: Number of results to return
            
        Returns:
            List of similar documents with scores
        """
        try:
            results = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=limit
            )
            
            return [
                {
                    "text": result.payload.get("text", ""),
                    "source": result.payload.get("source", ""),
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


if __name__ == "__main__":
    # Test the service
    service = QdrantService()
    service.create_collection()
    
    # Add sample documents
    docs = [
        {"text": "Machine learning is about data analysis", "source": "test.txt"},
        {"text": "Deep learning uses neural networks", "source": "test.txt"}
    ]
    embeddings = [[0.1] * 384, [0.2] * 384]  # Dummy embeddings
    service.add_documents(docs, embeddings)
    
    info = service.get_collection_info()
    print(f"Collection info: {info}")