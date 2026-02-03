import logging
from functools import cached_property

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str = "intfloat/multilingual-e5-base"):
        self._model_name = model_name
        self._model: SentenceTransformer | None = None

    @cached_property
    def model(self) -> SentenceTransformer:
        logger.info(f"Loading embedding model: {self._model_name}")
        return SentenceTransformer(self._model_name)

    def warmup(self) -> None:
        _ = self.model
        logger.info("Embedding model warmed up")

    def encode(self, texts: str | list[str]) -> np.ndarray:
        return self.model.encode(texts, convert_to_numpy=True)

    def cosine_similarity(
        self, query_embedding: np.ndarray, embeddings: np.ndarray
    ) -> np.ndarray:
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        embeddings_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        return np.dot(embeddings_norm, query_norm)
