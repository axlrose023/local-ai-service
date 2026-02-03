
import logging
from abc import ABC, abstractmethod

from ..models.document import SearchResult

logger = logging.getLogger(__name__)


class ScoringStrategy(ABC):
    """Base class for scoring strategies."""

    @abstractmethod
    def apply(self, query: str, results: list[SearchResult]) -> list[SearchResult]:
        """Apply strategy to results."""
        ...


class ScoreCutoffStrategy(ScoringStrategy):
    """Filter results with score much lower than top-1."""

    def __init__(self, score_ratio: float = 0.3):
        """Initialize strategy.

        Args:
            score_ratio: Minimum ratio of score to max_score.
        """
        self._score_ratio = score_ratio

    def apply(self, query: str, results: list[SearchResult]) -> list[SearchResult]:
        """Filter results below threshold."""
        if not results:
            return results

        max_score = results[0].score
        min_score = max_score * self._score_ratio

        filtered = [r for r in results if r.score >= min_score]

        if len(filtered) < len(results):
            logger.info(
                f"Score cutoff: {len(results)} → {len(filtered)} "
                f"(max={max_score:.2f}, min_allowed={min_score:.2f})"
            )

        return filtered


class KeywordBoostStrategy(ScoringStrategy):
    """Boost documents matching keywords in filename."""

    DEFAULT_KEYWORD_MAP = {
        "принтер": "принтер",
        "друк": "принтер",
        "відряджен": "відрядження",
        "travel": "відрядження",
        "добов": "відрядження",
        "кав": "кава",
        "кухн": "кухня",
        "холодильник": "кухня",
        "молок": "кухня",
        "wifi": "мереж",
        "vpn": "мереж",
        "пароль": "облікового",
        "пароля": "облікового",
        "безпек": "безпека",
        "флешк": "безпека",
        "usb": "безпека",
        "носі": "безпека",
        "конфіденц": "безпека",
        "jira": "jira",
        "задач": "jira",
        "slack": "етик",
        "пошт": "етик",
        "листуван": "етик",
        "гібрид": "віддален",
        "віддален": "віддален",
        "бонус": "бонус",
        "реферал": "бонус",
        "відпустк": "vacation",
        "vacation": "vacation",
    }

    def __init__(self, keyword_map: dict[str, str] | None = None):
        """Initialize strategy.

        Args:
            keyword_map: Custom keyword -> pattern mapping.
        """
        self._keyword_map = keyword_map or self.DEFAULT_KEYWORD_MAP

    def apply(self, query: str, results: list[SearchResult]) -> list[SearchResult]:
        """Boost documents matching query keywords."""
        if not results:
            return results

        query_lower = query.lower()
        boost_patterns = set()

        for keyword, pattern in self._keyword_map.items():
            if keyword in query_lower:
                boost_patterns.add(pattern)

        if not boost_patterns:
            return results

        boosted = []
        others = []

        for result in results:
            source_lower = result.source.lower()
            if any(p in source_lower for p in boost_patterns):
                boosted.append(result)
            else:
                others.append(result)

        if boosted:
            logger.info(
                f"Keyword boost: {len(boosted)} docs boosted (patterns: {boost_patterns})"
            )

        return boosted + others
