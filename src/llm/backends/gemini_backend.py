from __future__ import annotations

from .base import BackendError, LLMBackend, LLMResult


class GeminiBackend(LLMBackend):
    """Gemini-ready backend shell.

    This class intentionally keeps the same interface as HuggingFaceBackend,
    so provider swaps are additive and do not require pipeline refactors.
    """

    def __init__(self, *, api_key: str | None = None, model: str | None = None) -> None:
        self.api_key = api_key
        self.model = model or "gemini-2.5-pro"

    def score_document(
        self,
        text: str,
        *,
        ticker: str,
        doc_type: str,
        period: str,
    ) -> LLMResult:
        raise BackendError(
            "Gemini backend scaffold is present but not implemented yet. "
            "Use provider 'huggingface' now, then implement this method for Gemini API calls."
        )
