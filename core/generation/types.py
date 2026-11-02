from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional


class GenerationKind(str, Enum):
    NONE = "none"
    TRAINING_PLAN = "training_plan"
    EXERCISE_IDEATION = "exercise_ideation"
    LEGAL_DOCUMENT = "legal_document"


@dataclass
class ArtifactDraft:
    artifact_id: str
    kind: str
    template_path: Path
    draft_path: Path
    suggested_name: str
    preview_text: str
    can_save: bool = False
    validation_errors: List[str] = field(default_factory=list)


@dataclass
class AssistantResult:
    text: str
    generation_kind: GenerationKind = GenerationKind.NONE
    sources: List[str] = field(default_factory=list)
    clarification_fields: List[str] = field(default_factory=list)
    artifact: Optional[ArtifactDraft] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return self.text


@dataclass
class PendingGeneration:
    kind: GenerationKind
    original_request: str
    missing_fields: List[str]
    collected: Dict[str, str] = field(default_factory=dict)

