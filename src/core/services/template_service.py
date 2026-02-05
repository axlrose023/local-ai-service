"""Template service - LLM-driven template selection with semantic suggestion fallback."""

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
        suggest_threshold: float = 0.75,
    ):
        self._embedder = embedder
        self._templates_path = Path(templates_path)
        self._config_path = Path(config_path)
        self._suggest_threshold = suggest_threshold

        self._templates: list[TemplateInfo] = []
        self._templates_by_id: dict[str, TemplateInfo] = {}
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
                    id=item["id"],
                    file=item["file"],
                    name=item["name"],
                    description=item["description"],
                )
                self._templates.append(template)
                self._templates_by_id[template.id] = template

            logger.info(f"Loaded {len(self._templates)} templates from config")
        except Exception as e:
            logger.error(f"Failed to load templates config: {e}")

    def _compute_embeddings(self) -> None:
        if not self._templates:
            return

        for template in self._templates:
            text_for_embedding = f"{template.name}. {template.description}"
            embedding = self._embedder.encode(f"passage: {text_for_embedding}")
            self._template_embeddings[template.file] = embedding

        logger.info(
            f"Computed embeddings for {len(self._template_embeddings)} templates"
        )

    def get_by_id(self, template_id: str) -> Optional[TemplateMatch]:
        """Get template by LLM-assigned id (e.g. 'відпустка')."""
        template = self._templates_by_id.get(template_id)
        if not template:
            return None

        file_path = self._templates_path / template.file
        if not file_path.exists():
            logger.warning(f"Template file not found: {file_path}")
            return None

        return TemplateMatch(
            template=template,
            path=file_path,
            score=1.0,
            confidence=MatchConfidence.HIGH,
            matched_by="id",
        )

    def suggest_for_query(self, query: str) -> Optional[TemplateMatch]:
        """Suggest a template based on semantic similarity to user query.

        Used after docs-mode response to offer a related template.
        """
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

        if best_match and best_score >= self._suggest_threshold:
            file_path = self._templates_path / best_match.file
            if not file_path.exists():
                return None

            logger.info(
                f"Template suggestion: '{best_match.name}' "
                f"score={best_score:.2f} for query='{query[:60]}'"
            )
            return TemplateMatch(
                template=best_match,
                path=file_path,
                score=best_score,
                confidence=MatchConfidence.MEDIUM,
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
