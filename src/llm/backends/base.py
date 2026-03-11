from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Mapping


class BackendError(RuntimeError):
    """Raised when an LLM backend request fails or returns invalid output."""


@dataclass(slots=True)
class LLMResult:
    score: float
    evidence_summary: str
    confidence: float
    raw_response: str = ""
    metadata: Mapping[str, Any] | None = None


class LLMBackend(ABC):
    """Provider-agnostic scoring interface for document AI assessment."""

    @abstractmethod
    def score_document(
        self,
        text: str,
        *,
        ticker: str,
        doc_type: str,
        period: str,
    ) -> LLMResult:
        """Return a scored AI assessment for the provided document text."""
        raise NotImplementedError
