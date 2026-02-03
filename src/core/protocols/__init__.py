"""Protocol interfaces for dependency injection."""
from .embedder import EmbedderProtocol
from .vector_store import VectorStoreProtocol
from .reranker import RerankerProtocol
from .llm import LLMProtocol

__all__ = [
    "EmbedderProtocol",
    "VectorStoreProtocol",
    "RerankerProtocol",
    "LLMProtocol",
]
