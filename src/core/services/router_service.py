"""Router service - determines if RAG search is needed."""

import json
import logging
from pathlib import Path

import numpy as np

from ..models.chat import ChatHistory
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
        self._history_threshold = self._config.get("history_threshold", 0.55)
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
                "history_threshold": 0.55,
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

        # General knowledge patterns (e.g. "що таке X") without corporate
        # keywords — skip search and let LLM answer from general knowledge.
        for gp in self._config.get("general_patterns", []):
            if gp in text:
                self._log(f"General knowledge pattern: '{gp}' (no keyword)")
                return False

        # Avoid semantic routing for very short single-token queries without keywords.
        tokens = text.split()
        if len(tokens) == 1 and len(text) <= 6:
            self._log("Skip: single-token short query without keyword")
            return False

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

    def is_casual(self, query: str) -> bool:
        """Check if message is a greeting, thanks, or other casual text."""
        text = query.lower().strip()
        if len(text) < 3:
            return True
        for pattern in self._config.get("skip_patterns", []):
            if pattern in text:
                return True
        return False

    def is_related(self, query: str, previous: str) -> bool:
        """Check if two queries are semantically related."""
        q = query.strip()
        p = previous.strip()

        # Very short messages are likely follow-ups
        if len(q) < 3 or len(p) < 3:
            return True

        embeddings = self._embedder.encode([f"query: {q}", f"query: {p}"])
        score = float(self._embedder.cosine_similarity(embeddings[0], embeddings[1:])[0])
        decision = score >= self._history_threshold

        if self._debug:
            self._log(
                f"History relatedness: score={score:.3f} "
                f"threshold={self._history_threshold} -> {decision}"
            )

        return decision

    def _history_subset(self, history: ChatHistory, user_indices: list[int]) -> list[dict]:
        """Build a filtered history list with related user+assistant pairs."""
        keep: set[int] = set()
        for idx in user_indices:
            keep.add(idx)
            if idx + 1 < len(history.messages):
                if history.messages[idx + 1].role == "assistant":
                    keep.add(idx + 1)
        history_list = history.to_list()
        return [history_list[i] for i in range(len(history_list)) if i in keep]

    def select_history(self, query: str, history: ChatHistory) -> list[dict]:
        """Select relevant history based on semantic relatedness.

        Returns a filtered list of message dicts. If no relevant history is found,
        returns an empty list.
        """
        if not history.messages:
            return []

        if self.is_casual(query):
            self._log("History: skip for casual query")
            return []

        user_indices: list[int] = []
        user_texts: list[str] = []
        for i, msg in enumerate(history.messages):
            if msg.role != "user":
                continue
            if self.is_casual(msg.content):
                continue
            user_indices.append(i)
            user_texts.append(msg.content)

        if not user_texts:
            return []

        # Very short queries are often follow-ups; allow softer matching.
        is_short_query = len(query.strip().split()) <= 4

        embeddings = self._embedder.encode(
            [f"query: {query}"] + [f"query: {t}" for t in user_texts]
        )
        scores = self._embedder.cosine_similarity(embeddings[0], embeddings[1:])

        related_user_indices = [
            user_indices[i]
            for i, score in enumerate(scores)
            if score >= self._history_threshold
        ]

        if not related_user_indices:
            top_idx = int(np.argmax(scores))
            top_score = float(np.max(scores))
            soft_threshold = max(0.0, self._history_threshold - 0.1)
            if is_short_query and top_score >= soft_threshold:
                related_user_indices = [user_indices[top_idx]]
                if self._debug:
                    self._log(
                        f"History: soft match score={top_score:.3f} "
                        f"threshold={soft_threshold:.2f}"
                    )
            else:
                if self._debug:
                    self._log(
                        f"History: no related messages (top_score={top_score:.3f})"
                    )
                return []

        if self._debug:
            self._log(
                f"History: keeping {len(related_user_indices)} user message(s)"
            )

        return self._history_subset(history, related_user_indices)
