import logging

from sentence_transformers import CrossEncoder

from src.core.models.document import SearchResult

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """Reranker using CrossEncoder models."""

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        """Initialize reranker.

        Args:
            model_name: HuggingFace model name.
        """
        logger.info(f"Loading reranker: {model_name}")
        self._model = CrossEncoder(model_name)
        logger.info("Reranker loaded")

    def rerank(self, query: str, results: list[SearchResult]) -> list[SearchResult]:
        """Rerank results by relevance.

        Args:
            query: User query.
            results: Search results.

        Returns:
            Reranked results sorted by score (descending).
        """
        if not results:
            return results

        pairs = [[query, r.content] for r in results]
        scores = self._model.predict(pairs)

        for i, score in enumerate(scores):
            results[i].rerank_score = float(score)

        results.sort(key=lambda x: x.score, reverse=True)

        if logger.isEnabledFor(logging.DEBUG):
            top_scores = ", ".join(f"{r.score:.2f}" for r in results[:3])
            logger.debug(f"Reranker top-3 scores: [{top_scores}]")

        return results
