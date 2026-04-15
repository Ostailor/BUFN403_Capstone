from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

import requests
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import QWEN_MODEL_CANDIDATES
from .models import AskResult

log = logging.getLogger(__name__)


class QwenGenerationError(RuntimeError):
    pass


class QwenAnswerGenerator:
    def __init__(
        self,
        *,
        model_candidates: list[str] | None = None,
        hf_token: str | None = None,
        prefer_local: bool = True,
        local_files_only: bool = True,
        max_input_tokens: int = 4096,
        max_new_tokens: int = 500,
    ) -> None:
        self.model_candidates = model_candidates or QWEN_MODEL_CANDIDATES
        self.hf_token = hf_token or os.getenv("HF_TOKEN")
        self.prefer_local = prefer_local
        self.local_files_only = local_files_only
        self.max_input_tokens = max_input_tokens
        self.max_new_tokens = max_new_tokens
        self._tokenizer: Any | None = None
        self._model: Any | None = None
        self._active_model: str | None = None
        self._torch: Any | None = None
        self._device = "cpu"

    @property
    def active_model(self) -> str | None:
        return self._active_model

    def _load_local(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return
        import torch

        last_error: Exception | None = None
        if torch.cuda.is_available():
            self._device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self._device = "mps"
        else:
            self._device = "cpu"
        for candidate in self.model_candidates:
            try:
                model_kwargs: dict[str, Any] = {"local_files_only": self.local_files_only}
                if self._device == "cuda":
                    model_kwargs["dtype"] = torch.float16
                tokenizer = AutoTokenizer.from_pretrained(candidate, local_files_only=self.local_files_only)
                model = AutoModelForCausalLM.from_pretrained(candidate, **model_kwargs)
                model.to(self._device)
                model.eval()
                self._tokenizer = tokenizer
                self._model = model
                self._torch = torch
                self._active_model = candidate
                log.info("Loaded Qwen model %s on %s", candidate, self._device)
                return
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                self._tokenizer = None
                self._model = None
                continue
        raise QwenGenerationError(f"Unable to load any local Qwen model: {last_error}")

    def _build_prompt(self, question: str, evidence: str, structured_context: str) -> str:
        return (
            "You answer research questions about how banks say they are using AI.\n"
            "Use only the supplied evidence.\n"
            "If the evidence is weak or incomplete, say 'insufficient evidence'.\n"
            "Return JSON only with keys: answer_text, citations, theme_tags, confidence.\n"
            "Citations must reference the provided chunk ids or structured refs exactly.\n"
            f"Question: {question}\n"
            f"Narrative evidence:\n{evidence}\n"
            f"Structured evidence:\n{structured_context}\n"
        )

    def _extract_json(self, raw_text: str) -> dict[str, Any]:
        match = re.search(r"\{.*\}", raw_text, flags=re.S)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass

        partial_payload = self._extract_partial_json(raw_text)
        if partial_payload is not None:
            return partial_payload
        raise QwenGenerationError(f"No JSON payload found in response: {raw_text[:500]}")

    def _extract_partial_json(self, raw_text: str) -> dict[str, Any] | None:
        text = raw_text.strip()
        if "{" not in text:
            return None

        payload: dict[str, Any] = {}

        intent_match = re.search(r'"intent_level"\s*:\s*(\d+)', text)
        if intent_match:
            payload["intent_level"] = int(intent_match.group(1))

        label_match = re.search(r'"intent_label"\s*:\s*"([^"]*)"', text)
        if label_match:
            payload["intent_label"] = label_match.group(1)

        categories_match = re.search(r'"app_categories"\s*:\s*\[(.*?)\]', text, flags=re.S)
        if categories_match:
            payload["app_categories"] = re.findall(r'"([^"]*)"', categories_match.group(1))

        confidence_match = re.search(r'"confidence"\s*:\s*("([^"]*)"|[-+]?\d*\.?\d+)', text)
        if confidence_match:
            payload["confidence"] = confidence_match.group(2) or confidence_match.group(1)

        evidence_match = re.search(r'"evidence_snippet"\s*:\s*"((?:[^"\\]|\\.)*)', text, flags=re.S)
        if evidence_match:
            snippet = evidence_match.group(1)
            payload["evidence_snippet"] = bytes(snippet, "utf-8").decode("unicode_escape", errors="ignore")

        required_keys = {"intent_level", "intent_label"}
        if required_keys.issubset(payload):
            payload.setdefault("app_categories", [])
            payload.setdefault("confidence", 0.0)
            payload.setdefault("evidence_snippet", "")
            return payload
        return None

    def _render_messages(self, messages: list[dict[str, str]]) -> str:
        assert self._tokenizer is not None
        if getattr(self._tokenizer, "chat_template", None):
            return self._tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return "\n".join(f"{message['role']}: {message['content']}" for message in messages)

    def _generate_local_messages(
        self,
        messages: list[dict[str, str]],
        *,
        max_new_tokens: int | None = None,
    ) -> dict[str, Any]:
        self._load_local()
        assert self._model is not None
        assert self._tokenizer is not None
        assert self._torch is not None

        rendered = self._render_messages(messages)
        encoded = self._tokenizer(
            rendered,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_tokens,
        )
        device = next(self._model.parameters()).device
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with self._torch.no_grad():
            generated = self._model.generate(
                **encoded,
                max_new_tokens=max_new_tokens or self.max_new_tokens,
                do_sample=False,
                pad_token_id=self._tokenizer.eos_token_id,
                eos_token_id=self._tokenizer.eos_token_id,
            )
        prompt_length = encoded["input_ids"].shape[-1]
        raw_text = self._tokenizer.decode(generated[0][prompt_length:], skip_special_tokens=True)
        return self._extract_json(raw_text)

    def _generate_remote_messages(
        self,
        messages: list[dict[str, str]],
        *,
        max_new_tokens: int | None = None,
    ) -> dict[str, Any]:
        if not self.hf_token:
            raise QwenGenerationError("HF_TOKEN is required for remote generation")
        response = requests.post(
            "https://router.huggingface.co/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {self.hf_token}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model_candidates[0],
                "messages": messages,
                "temperature": 0.1,
                "max_tokens": max_new_tokens or self.max_new_tokens,
            },
            timeout=90,
        )
        response.raise_for_status()
        payload = response.json()
        raw_text = payload["choices"][0]["message"]["content"]
        self._active_model = self.model_candidates[0]
        return self._extract_json(raw_text)

    def generate_json(
        self,
        *,
        messages: list[dict[str, str]],
        prefer_local: bool | None = None,
        max_new_tokens: int | None = None,
    ) -> dict[str, Any]:
        use_local = self.prefer_local if prefer_local is None else prefer_local
        if use_local:
            try:
                return self._generate_local_messages(messages, max_new_tokens=max_new_tokens)
            except Exception:
                if self.hf_token:
                    return self._generate_remote_messages(messages, max_new_tokens=max_new_tokens)
                raise
        return self._generate_remote_messages(messages, max_new_tokens=max_new_tokens)

    def answer(
        self,
        *,
        question: str,
        evidence_blocks: list[tuple[str, str]],
        structured_refs: list[dict[str, Any]],
        prompt_hint: str = "",
    ) -> AskResult:
        if not evidence_blocks and not structured_refs:
            return AskResult(
                answer_text="insufficient evidence",
                citations=[],
                retrieved_chunk_ids=[],
                structured_data_refs=[],
                theme_tags=[],
                confidence=0.0,
                metadata={"model": "none"},
            )
        evidence = "\n\n".join(f"{chunk_id}: {text}" for chunk_id, text in evidence_blocks)
        structured_context = json.dumps(structured_refs, indent=2, sort_keys=True)
        full_question = f"{prompt_hint}\n\n{question}".strip() if prompt_hint else question
        prompt = self._build_prompt(question=full_question, evidence=evidence, structured_context=structured_context)
        messages = [
            {"role": "system", "content": "Respond with one JSON object and no markdown."},
            {"role": "user", "content": prompt},
        ]
        try:
            payload = self.generate_json(messages=messages)
        except Exception:
            payload = self._generate_remote_messages(messages) if self.hf_token else {
                "answer_text": evidence_blocks[0][1][:500] if evidence_blocks else "insufficient evidence",
                "citations": [chunk_id for chunk_id, _ in evidence_blocks[:3]],
                "theme_tags": [],
                "confidence": 0.2,
            }
        citations = [str(value) for value in payload.get("citations", [])]
        answer_text = str(payload.get("answer_text", "")).strip() or "insufficient evidence"
        theme_tags = [str(value) for value in payload.get("theme_tags", [])]
        confidence = float(payload.get("confidence", 0.2))
        return AskResult(
            answer_text=answer_text,
            citations=citations,
            retrieved_chunk_ids=[chunk_id for chunk_id, _ in evidence_blocks],
            structured_data_refs=structured_refs,
            theme_tags=theme_tags,
            confidence=max(0.0, min(confidence, 1.0)),
            metadata={"model": self.active_model or "extractive-fallback", "device": self._device},
        )
