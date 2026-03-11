"""Backend implementations for model providers."""

from .base import BackendError, LLMBackend, LLMResult
from .gemini_backend import GeminiBackend
from .huggingface_backend import HuggingFaceBackend

__all__ = [
    "BackendError",
    "LLMBackend",
    "LLMResult",
    "GeminiBackend",
    "HuggingFaceBackend",
]
