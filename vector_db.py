"""
Qdrant vector storage wrapper for RAG system.

Handles:
- Collection creation
- Vector upserts
- Similarity search
"""

from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct


class QdrantStorage:
    """
    Abstraction layer over QdrantClient for vector operations.
    """

    def __init__(
        self,
        url: str = "http://localhost:6333",
        collection: str = "docs",
        dim: int = 768,
    ):
        """
        Initialize Qdrant storage and ensure collection exists.

        Args:
            url: Qdrant server URL
            collection: Collection name
            dim: Embedding vector dimension
        """

        self.client = QdrantClient(url=url, timeout=30)
        self.collection = collection
        self.dim = dim

        if not self.client.collection_exists(collection_name=self.collection):
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(
                    size=self.dim,
                    distance=Distance.COSINE,
                ),
            )

    # ------------------------------------------------------------------
    # UPSERT
    # ------------------------------------------------------------------

    def upsert(self, ids: list[str], vectors: list[list[float]], payloads: list[dict]):
        """
        Insert or update vector points in Qdrant.

        Args:
            ids: Unique identifiers
            vectors: Embedding vectors
            payloads: Metadata payloads
        """

        points = [
            PointStruct(
                id=ids[i],
                vector=vectors[i],
                payload=payloads[i],
            )
            for i in range(len(ids))
        ]

        self.client.upsert(
            collection_name=self.collection,
            points=points,
        )

    # ------------------------------------------------------------------
    # SEARCH (UPDATED FOR NEW QDRANT API)
    # ------------------------------------------------------------------

    def search(self, query_vector: list[float], top_k: int = 5):
        """
        Perform similarity search using cosine distance.

        Args:
            query_vector: Embedded query vector
            top_k: Number of results to return

        Returns:
            dict with:
                - context: List[str]
                - sources: List[str]
        """

        results = self.client.query_points(
            collection_name=self.collection,
            query=query_vector,
            limit=top_k,
            with_payload=True,
        )

        context = []
        sources = set()

        for point in results.points:
            payload = point.payload or {}

            text = payload.get("text", "")
            source = payload.get("source", "")

            if text:
                context.append(text)

            if source:
                sources.add(source)

        return {
            "contexts": context,
            "sources": list(sources),
        }