"""Document domain models."""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Chunk:
    """Document chunk for indexing."""
    content: str
    source: str
    file_path: str
    file_hash: str
    chunk_index: int


@dataclass
class SearchResult:
    """Search result from vector store."""
    content: str
    source: str
    vector_score: float
    rerank_score: Optional[float] = None

    @property
    def score(self) -> float:
        """Final score (rerank if available, else vector)."""
        return self.rerank_score if self.rerank_score is not None else self.vector_score


@dataclass
class SearchResponse:
    """Search response for presentation layer."""
    results: list[SearchResult]
    context: str
    sources: list[str]
