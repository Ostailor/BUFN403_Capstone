from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable

from .config import EMBEDDING_MODEL_CANDIDATES


class EmbedderError(RuntimeError):
    pass


@dataclass(slots=True)
class HashingEmbedder:
    dimension: int = 96

    def encode(self, texts: Iterable[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            values = []
            for index in range(self.dimension):
                byte = digest[index % len(digest)]
                values.append((byte / 255.0) * 2.0 - 1.0)
            vectors.append(values)
        return vectors


class TransformerMeanPoolEmbedder:
    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or EMBEDDING_MODEL_CANDIDATES[0]
        self._model: object | None = None
        self._tokenizer: object | None = None
        self._torch: object | None = None

    def _load(self) -> object:
        if self._model is None or self._tokenizer is None or self._torch is None:
            try:
                import torch
                from transformers import AutoModel, AutoTokenizer
            except Exception as exc:  # noqa: BLE001
                raise EmbedderError("transformers/torch could not be imported for embeddings") from exc
            last_error: Exception | None = None
            for candidate in [self.model_name, *EMBEDDING_MODEL_CANDIDATES[1:]]:
                try:
                    self._tokenizer = AutoTokenizer.from_pretrained(candidate)
                    self._model = AutoModel.from_pretrained(candidate)
                    self._torch = torch
                    self.model_name = candidate
                    break
                except Exception as exc:  # noqa: BLE001
                    last_error = exc
                    self._model = None
                    self._tokenizer = None
                    self._torch = None
                    continue
            if self._model is None:
                raise EmbedderError(f"Failed loading embedding model: {last_error}")
        return self._model

    def encode(self, texts: Iterable[str]) -> list[list[float]]:
        model = self._load()
        assert self._tokenizer is not None
        assert self._torch is not None
        texts_list = list(texts)
        if not texts_list:
            return []
        encoded = self._tokenizer(
            texts_list,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        with self._torch.no_grad():
            outputs = model(**encoded)
        attention_mask = encoded["attention_mask"]
        token_embeddings = outputs.last_hidden_state
        mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        summed = (token_embeddings * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        embeddings = summed / counts
        embeddings = self._torch.nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings.cpu().tolist()


def build_embedder(name: str | None) -> TransformerMeanPoolEmbedder | HashingEmbedder:
    if name == "hash":
        return HashingEmbedder()
    return TransformerMeanPoolEmbedder(model_name=name)
