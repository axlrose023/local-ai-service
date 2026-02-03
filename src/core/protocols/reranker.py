"""Reranker protocol for dependency injection."""
from typing import Protocol, runtime_checkable

from ..models.document import SearchResult


@runtime_checkable
class RerankerProtocol(Protocol):
    """Protocol for reranking service."""

    def rerank(
        self,
        query: str,
        results: list[SearchResult]
    ) -> list[SearchResult]:
        """Rerank search results by relevance.

        Args:
            query: User query.
            results: Search results to rerank.

        Returns:
            Reranked results sorted by relevance.
        """
        ...
