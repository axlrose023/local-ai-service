"""Router service - determines if RAG search is needed."""

import json
import logging
from pathlib import Path

import numpy as np

from ..protocols.embedder import EmbedderProtocol

logger = logging.getLogger(__name__)


class RouterService:
    """Hybrid router: skip patterns → keywords → semantic matching."""

    def __init__(
        self,
        embedder: EmbedderProtocol,
        config_path: str = "router_config.json",
        threshold: float = 0.78,
    ):
        """Initialize router.

        Args:
            embedder: Embedding service for semantic matching.
            config_path: Path to router config JSON.
            threshold: Semantic similarity threshold.
        """
        self._embedder = embedder
        self._debug = False
        self._config = self._load_config(config_path)
        self._threshold = self._config.get("threshold", threshold)
        self._debug = self._config.get("debug", False)

        anchors = self._config.get("semantic_anchors", [])
        if anchors:
            self._log("Vectorizing semantic anchors...")
            anchors_with_prefix = [f"passage: {a}" for a in anchors]
            self._anchor_embeddings = self._embedder.encode(anchors_with_prefix)
            self._log(f"Loaded {len(anchors)} anchors")
        else:
            self._anchor_embeddings = None

    def _load_config(self, path: str) -> dict:
        """Load config from JSON."""
        config_file = Path(path)
        if not config_file.exists():
            logger.warning(f"Router config {path} not found, using defaults")
            return {
                "skip_patterns": ["привіт", "дякую", "бувай"],
                "keywords": ["vpn", "відпустка", "пароль"],
                "semantic_anchors": [],
                "threshold": 0.78,
                "debug": False,
            }

        with open(config_file, "r", encoding="utf-8") as f:
            config = json.load(f)
            self._log(f"Config loaded from {path}")
            return config

    def _log(self, message: str) -> None:
        """Log debug message."""
        if self._debug:
            logger.info(f"[router] {message}")

    def should_search(self, query: str) -> bool:
        """Determine if RAG search is needed.

        Args:
            query: User query.

        Returns:
            True if search needed, False otherwise.
        """
        text = query.lower().strip()

        if len(text) < 3:
            self._log(f"Skip: too short ({len(text)} chars)")
            return False

        for pattern in self._config.get("skip_patterns", []):
            if pattern in text:
                self._log(f"Skip pattern: '{pattern}'")
                return False

        for keyword in self._config.get("keywords", []):
            if keyword in text:
                self._log(f"Keyword match: '{keyword}'")
                return True

        if self._anchor_embeddings is not None:
            query_embedding = self._embedder.encode(f"query: {query}")
            scores = self._embedder.cosine_similarity(
                query_embedding, self._anchor_embeddings
            )
            max_score = float(np.max(scores))
            max_idx = int(np.argmax(scores))

            decision = max_score > self._threshold

            if self._debug:
                anchor = self._config["semantic_anchors"][max_idx]
                self._log(
                    f"Semantic: score={max_score:.3f} threshold={self._threshold} "
                    f"-> {decision} (best: '{anchor[:50]}...')"
                )

            return decision

        self._log("Fallback: no semantic anchors")
        return False
