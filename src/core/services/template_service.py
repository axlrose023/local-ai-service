"""Template service - hybrid template matching with triggers and semantic search."""

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np

from src.core.models.template import MatchConfidence, TemplateInfo, TemplateMatch
from src.core.protocols.embedder import EmbedderProtocol

logger = logging.getLogger(__name__)


class TemplateService:

    def __init__(
        self,
        embedder: EmbedderProtocol,
        templates_path: str = "./templates",
        config_path: str = "templates_config.json",
        high_threshold: float = 0.85,
        low_threshold: float = 0.70,
    ):
        self._embedder = embedder
        self._templates_path = Path(templates_path)
        self._config_path = Path(config_path)
        self._high_threshold = high_threshold
        self._low_threshold = low_threshold

        self._templates: list[TemplateInfo] = []
        self._template_embeddings: dict[str, np.ndarray] = {}

        self._load_config()
        self._compute_embeddings()

    def _load_config(self) -> None:
        if not self._config_path.exists():
            logger.warning(f"Templates config not found: {self._config_path}")
            return

        try:
            with open(self._config_path, "r", encoding="utf-8") as f:
                config = json.load(f)

            for item in config.get("templates", []):
                template = TemplateInfo(
                    file=item["file"],
                    name=item["name"],
                    description=item["description"],
                    triggers=item.get("triggers", []),
                )
                self._templates.append(template)

            logger.info(f"Loaded {len(self._templates)} templates from config")
        except Exception as e:
            logger.error(f"Failed to load templates config: {e}")

    def _compute_embeddings(self) -> None:
        if not self._templates:
            return

        for template in self._templates:
            text_for_embedding = (
                f"{template.name}. {template.description}. {' '.join(template.triggers)}"
            )
            embedding = self._embedder.encode(f"passage: {text_for_embedding}")
            self._template_embeddings[template.file] = embedding

        logger.info(
            f"Computed embeddings for {len(self._template_embeddings)} templates"
        )

    def _match_triggers(self, query: str) -> Optional[tuple[TemplateInfo, float]]:
        query_lower = query.lower()
        query_words = set(query_lower.split())

        best_match: Optional[TemplateInfo] = None
        best_score = 0.0

        for template in self._templates:
            for trigger in template.triggers:
                trigger_lower = trigger.lower()
                trigger_words = set(trigger_lower.split())

                if trigger_lower in query_lower:
                    base_score = len(trigger_lower) / max(len(query_lower), 1)
                    word_bonus = min(0.3, len(trigger_words) * 0.1)
                    coverage_bonus = 0.2 if base_score > 0.5 else 0.0
                    score = min(1.0, base_score + word_bonus + coverage_bonus)

                    if score > best_score:
                        best_score = score
                        best_match = template

                elif trigger_words & query_words:
                    overlap = len(trigger_words & query_words)
                    total = len(trigger_words | query_words)
                    score = overlap / total * 0.6

                    if score > best_score:
                        best_score = score
                        best_match = template

        if best_match and best_score > 0.3:
            return (best_match, best_score)
        return None

    def _match_semantic(self, query: str) -> Optional[tuple[TemplateInfo, float]]:
        if not self._template_embeddings:
            return None

        query_embedding = self._embedder.encode(f"query: {query}")

        best_match: Optional[TemplateInfo] = None
        best_score = 0.0

        for template in self._templates:
            template_embedding = self._template_embeddings.get(template.file)
            if template_embedding is None:
                continue

            dot_product = np.dot(query_embedding, template_embedding)
            norm_product = np.linalg.norm(query_embedding) * np.linalg.norm(
                template_embedding
            )
            similarity = dot_product / norm_product if norm_product > 0 else 0.0

            if similarity > best_score:
                best_score = float(similarity)
                best_match = template

        if best_match:
            return (best_match, best_score)
        return None

    def _get_confidence(self, score: float) -> MatchConfidence:
        if score >= self._high_threshold:
            return MatchConfidence.HIGH
        elif score >= self._low_threshold:
            return MatchConfidence.MEDIUM
        else:
            return MatchConfidence.LOW

    def match(self, query: str) -> Optional[TemplateMatch]:
        if not self._templates:
            return None

        trigger_result = self._match_triggers(query)
        if trigger_result:
            template, score = trigger_result
            boosted_score = min(1.0, score + 0.15)
            confidence = self._get_confidence(boosted_score)

            if confidence != MatchConfidence.LOW:
                file_path = self._templates_path / template.file
                logger.info(
                    f"Trigger match: '{template.name}' "
                    f"score={boosted_score:.2f} ({confidence.value})"
                )
                return TemplateMatch(
                    template=template,
                    path=file_path,
                    score=boosted_score,
                    confidence=confidence,
                    matched_by="trigger",
                )

        semantic_result = self._match_semantic(query)
        if semantic_result:
            template, score = semantic_result
            confidence = self._get_confidence(score)

            if confidence != MatchConfidence.LOW:
                file_path = self._templates_path / template.file
                logger.info(
                    f"Semantic match: '{template.name}' "
                    f"score={score:.2f} ({confidence.value})"
                )
                return TemplateMatch(
                    template=template,
                    path=file_path,
                    score=score,
                    confidence=confidence,
                    matched_by="semantic",
                )

        return None

    def list_templates(self) -> list[TemplateInfo]:
        return self._templates.copy()

    def get_template_path(self, file_name: str) -> Optional[Path]:
        path = self._templates_path / file_name
        if path.exists():
            return path
        return None
