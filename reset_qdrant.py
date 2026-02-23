"""
Utility script to delete an existing Qdrant collection.

Use this only when you change embedding dimensions.
"""

from qdrant_client import QdrantClient

COLLECTION_NAME = "docs"

client = QdrantClient(url="http://localhost:6333")

if client.collection_exists(COLLECTION_NAME):
    client.delete_collection(COLLECTION_NAME)
    print(f"Collection '{COLLECTION_NAME}' deleted successfully.")
else:
    print(f"Collection '{COLLECTION_NAME}' does not exist.")