"""Search service - core RAG search logic."""

import logging
from typing import Optional

from ..models.document import SearchResponse, SearchResult
from ..protocols.embedder import EmbedderProtocol
from ..protocols.reranker import RerankerProtocol
from ..protocols.vector_store import VectorStoreProtocol
from ..strategies.scoring import (
    KeywordBoostStrategy,
    ScoreCutoffStrategy,
    ScoringStrategy,
)

logger = logging.getLogger(__name__)


class SearchService:
    """Search service with reranking and filtering strategies."""

    def __init__(
        self,
        embedder: EmbedderProtocol,
        vector_store: VectorStoreProtocol,
        reranker: RerankerProtocol,
        top_k: int = 5,
        fetch_k: int = 20,
        vector_threshold: float = 0.3,
        reranker_threshold: float = 0.0,
        score_ratio: float = 0.3,
        strategies: list[ScoringStrategy] | None = None,
    ):
        """Initialize search service.

        Args:
            embedder: Embedding service.
            vector_store: Vector store.
            reranker: Reranking service.
            top_k: Number of results to return.
            fetch_k: Number of candidates to fetch for reranking.
            vector_threshold: Minimum vector similarity.
            reranker_threshold: Minimum reranker score.
            score_ratio: Minimum score ratio for cutoff.
            strategies: Custom scoring strategies.
        """
        self._embedder = embedder
        self._vector_store = vector_store
        self._reranker = reranker
        self._top_k = top_k
        self._fetch_k = fetch_k
        self._vector_threshold = vector_threshold
        self._reranker_threshold = reranker_threshold

        self._strategies = strategies or [
            ScoreCutoffStrategy(score_ratio),
            KeywordBoostStrategy(),
        ]

    def search(self, query: str, top_k: Optional[int] = None) -> SearchResponse:
        """Search documents with reranking.

        Args:
            query: Search query.
            top_k: Override number of results.

        Returns:
            Search response with results, context, and sources.
        """
        top_k = top_k or self._top_k

        try:
            query_embedding = self._embedder.encode(f"query: {query}").tolist()

            results = self._vector_store.query(
                query_embedding=query_embedding, n_results=self._fetch_k
            )

            results = [r for r in results if r.vector_score >= self._vector_threshold]

            if results:
                results = self._reranker.rerank(query, results)

            results = [r for r in results if r.score > self._reranker_threshold]

            for strategy in self._strategies:
                results = strategy.apply(query, results)

            results = results[:top_k]

            logger.info(
                f"Search: returned {len(results)}/{top_k} docs for '{query[:50]}...'"
            )

            return SearchResponse(
                results=results,
                context=self._format_context(results),
                sources=self._get_unique_sources(results),
            )

        except Exception as e:
            logger.error(f"Search error: {e}")
            return SearchResponse(results=[], context="", sources=[])

    def _format_context(self, results: list[SearchResult]) -> str:
        """Format results as context for LLM."""
        if not results:
            return ""

        parts = []
        for i, r in enumerate(results, 1):
            source_name = r.source.rsplit(".", 1)[0].replace("_", " ").lstrip("0123456789 ")
            parts.append(f"[{i}] {source_name}:\n{r.content}")

        return "\n\n".join(parts)

    def _get_unique_sources(self, results: list[SearchResult]) -> list[str]:
        """Get unique source filenames."""
        seen = set()
        sources = []
        for r in results:
            if r.source not in seen:
                seen.add(r.source)
                sources.append(r.source)
        return sources
