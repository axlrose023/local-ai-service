"""Template domain models."""
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional


class MatchConfidence(Enum):
    """Confidence level for template match."""
    HIGH = "high"      # > high_threshold: deliver file immediately
    MEDIUM = "medium"  # between thresholds: ask for confirmation
    LOW = "low"        # < low_threshold: not relevant


@dataclass
class TemplateInfo:
    """Template metadata from config."""
    id: str
    file: str
    name: str
    description: str


@dataclass
class TemplateMatch:
    """Result of template matching."""
    template: TemplateInfo
    path: Path
    score: float
    confidence: MatchConfidence
    matched_by: str  # "id" or "semantic"

    @property
    def file_name(self) -> str:
        return self.template.file

    @property
    def display_name(self) -> str:
        return self.template.name
