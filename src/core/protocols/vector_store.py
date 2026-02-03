"""Vector store protocol for dependency injection."""
from typing import Protocol, runtime_checkable

from ..models.document import SearchResult


@runtime_checkable
class VectorStoreProtocol(Protocol):
    """Protocol for vector storage."""

    def add(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict]
    ) -> None:
        """Add documents to the store.

        Args:
            ids: Document IDs.
            embeddings: Document embeddings.
            documents: Document texts.
            metadatas: Document metadata.
        """
        ...

    def query(
        self,
        query_embedding: list[float],
        n_results: int = 5
    ) -> list[SearchResult]:
        """Search by embedding.

        Args:
            query_embedding: Query vector.
            n_results: Number of results to return.

        Returns:
            List of search results.
        """
        ...

    def count(self) -> int:
        """Get document count."""
        ...

    def get_all_metadatas(self) -> list[dict]:
        """Get all document metadatas."""
        ...
