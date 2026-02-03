import logging
from typing import Optional

import requests

from src.core.models.document import SearchResult

logger = logging.getLogger(__name__)


class ChromaVectorStore:
    """Vector store using ChromaDB HTTP API."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 8001,
        collection_name: str = "documents",
        tenant: str = "default_tenant",
        database: str = "default_database",
    ):
        """Initialize ChromaDB client.

        Args:
            host: ChromaDB host.
            port: ChromaDB port.
            collection_name: Collection name.
            tenant: Tenant name.
            database: Database name.
        """
        self._base_url = f"http://{host}:{port}/api/v2"
        self._tenant = tenant
        self._database = database
        self._collection_name = collection_name
        self._collection_id: Optional[str] = None

    @property
    def _collections_url(self) -> str:
        return f"{self._base_url}/tenants/{self._tenant}/databases/{self._database}/collections"

    def _ensure_collection(self) -> str:
        """Get or create collection, return ID."""
        if self._collection_id:
            return self._collection_id

        resp = requests.get(self._collections_url)
        if resp.status_code == 200:
            for col in resp.json():
                if col["name"] == self._collection_name:
                    self._collection_id = col["id"]
                    return self._collection_id

        resp = requests.post(
            self._collections_url,
            json={"name": self._collection_name, "metadata": {"hnsw:space": "cosine"}},
        )
        resp.raise_for_status()
        self._collection_id = resp.json()["id"]
        logger.info(f"Created collection: {self._collection_name}")
        return self._collection_id

    def add(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict],
    ) -> None:
        """Add documents to collection."""
        col_id = self._ensure_collection()
        resp = requests.post(
            f"{self._collections_url}/{col_id}/add",
            json={
                "ids": ids,
                "embeddings": embeddings,
                "documents": documents,
                "metadatas": metadatas,
            },
        )
        resp.raise_for_status()

    def query(
        self, query_embedding: list[float], n_results: int = 5
    ) -> list[SearchResult]:
        """Search by embedding."""
        col_id = self._ensure_collection()
        resp = requests.post(
            f"{self._collections_url}/{col_id}/query",
            json={
                "query_embeddings": [query_embedding],
                "n_results": n_results,
                "include": ["documents", "metadatas", "distances"],
            },
        )

        if resp.status_code != 200:
            return []

        data = resp.json()
        results = []

        if data.get("ids") and data["ids"][0]:
            for i in range(len(data["ids"][0])):
                distance = data["distances"][0][i]
                similarity = 1.0 - distance

                results.append(
                    SearchResult(
                        content=data["documents"][0][i],
                        source=data["metadatas"][0][i].get("source", "Unknown"),
                        vector_score=similarity,
                    )
                )

        return results

    def count(self) -> int:
        """Get document count."""
        col_id = self._ensure_collection()
        resp = requests.get(f"{self._collections_url}/{col_id}/count")
        return resp.json() if resp.status_code == 200 else 0

    def get_all_metadatas(self) -> list[dict]:
        """Get all document metadatas."""
        col_id = self._ensure_collection()
        resp = requests.post(
            f"{self._collections_url}/{col_id}/get", json={"include": ["metadatas"]}
        )
        if resp.status_code == 200:
            return resp.json().get("metadatas", [])
        return []
