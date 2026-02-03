"""Embedder protocol for dependency injection."""
from typing import Protocol, runtime_checkable
import numpy as np


@runtime_checkable
class EmbedderProtocol(Protocol):
    """Protocol for embedding service."""

    def encode(self, texts: str | list[str]) -> np.ndarray:
        """Encode text(s) to embeddings.

        Args:
            texts: Single text or list of texts to encode.

        Returns:
            Numpy array of embeddings.
        """
        ...

    def cosine_similarity(
        self,
        query_embedding: np.ndarray,
        embeddings: np.ndarray
    ) -> np.ndarray:
        """Compute cosine similarity between query and embeddings.

        Args:
            query_embedding: Query embedding vector.
            embeddings: Matrix of embeddings to compare against.

        Returns:
            Array of similarity scores.
        """
        ...

    def warmup(self) -> None:
        """Pre-load the model for faster inference."""
        ...
