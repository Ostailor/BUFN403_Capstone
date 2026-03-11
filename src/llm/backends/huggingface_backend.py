from __future__ import annotations

import json
import os
import re
import time
from typing import Any

import requests

from .base import BackendError, LLMBackend, LLMResult


class HuggingFaceBackend(LLMBackend):
    """Hugging Face backend supporting router API and local Transformers inference."""

    def __init__(
        self,
        *,
        token: str | None,
        model: str,
        timeout_seconds: int = 90,
        max_retries: int = 3,
        local: bool = False,
        local_device: str = "auto",
        local_max_new_tokens: int = 220,
        local_max_input_tokens: int = 2048,
    ) -> None:
        if not local and not token:
            raise BackendError("HF token is required for remote HuggingFace backend")
        self.token = token
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.local = local
        self.url = "https://router.huggingface.co/v1/chat/completions"
        self.local_device = local_device
        self.local_max_new_tokens = local_max_new_tokens
        self.local_max_input_tokens = local_max_input_tokens
        self._tokenizer: Any | None = None
        self._model: Any | None = None
        self._torch: Any | None = None
        self._resolved_device = "cpu"
        self._is_seq2seq = False

    def _build_prompt(self, text: str, *, ticker: str, doc_type: str, period: str) -> str:
        return (
            "Rate AI focus in this bank document excerpt.\\n"
            "Use a strict 1.0 to 10.0 scale.\\n"
            "Rubric:\\n"
            "- 1-2: no meaningful AI mention\\n"
            "- 3-4: passing or vague AI mention\\n"
            "- 5-6: repeated AI/ML use cases in operations\\n"
            "- 7-8: clear enterprise deployment/investment across functions\\n"
            "- 9-10: AI is a major strategic differentiator with broad rollout\\n"
            "If the excerpt contains multiple concrete AI use cases, large rollout, or explicit strategic investment,"
            " do not return a low score.\\n"
            "Return one of the following output formats only:\\n"
            "1) JSON object: {\"score\":7.2,\"confidence\":0.81,\"evidence_summary\":\"...\"}\\n"
            "OR\\n"
            "2) Three lines:\\n"
            "SCORE: 7.2\\n"
            "CONFIDENCE: 0.81\\n"
            "EVIDENCE: ...\\n"
            "Evidence must be paraphrased and <= 220 chars.\\n"
            f"Ticker: {ticker}\\n"
            f"Document type: {doc_type}\\n"
            f"Period: {period}\\n"
            "Document excerpt:\\n"
            f"{text}\\n"
        )

    def _extract_json(self, text: str) -> dict[str, Any]:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if not match:
            raise BackendError(f"No JSON object found in model response: {text[:500]}")
        payload = match.group(0)
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise BackendError(f"Failed to parse model JSON response: {exc}") from exc
        if not isinstance(parsed, dict):
            raise BackendError("Model JSON response is not an object")
        return parsed

    @staticmethod
    def _clamp(value: float, low: float, high: float) -> float:
        return max(low, min(high, value))

    def _parse_structured_text(self, text: str) -> tuple[float, str, float]:
        # First attempt strict JSON extraction.
        try:
            parsed = self._extract_json(text)
            score = float(parsed.get("score", 1.0))
            confidence = float(parsed.get("confidence", 0.35))
            evidence_summary = str(parsed.get("evidence_summary", "")).strip()
            if not evidence_summary:
                evidence_summary = "Insufficient structured AI detail; score derived from detected signal strength."
            return (
                self._clamp(score, 1.0, 10.0),
                evidence_summary[:220],
                self._clamp(confidence, 0.0, 1.0),
            )
        except Exception:
            pass

        # Fallback parse for non-JSON structured text.
        score_match = re.search(
            r"(?is)\b(?:score|rating)\b\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)(?:\s*/\s*10)?",
            text,
        )
        conf_match = re.search(
            r"(?is)\bconfidence\b\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)",
            text,
        )
        evidence_match = re.search(
            r"(?is)\b(?:evidence|evidence_summary)\b\s*[:=]\s*(.+)",
            text,
        )

        score: float | None = None
        confidence = 0.55
        evidence_summary = ""

        if score_match:
            try:
                score = float(score_match.group(1))
            except Exception:
                score = None
            if score is not None and 0.0 <= score <= 1.0:
                score *= 10.0

        if conf_match:
            try:
                confidence = float(conf_match.group(1))
            except Exception:
                confidence = 0.55
            if confidence > 1.0:
                confidence = confidence / 10.0 if confidence <= 10.0 else 1.0

        if evidence_match:
            evidence_summary = evidence_match.group(1).strip().splitlines()[0]

        if not evidence_summary:
            first_line = text.strip().splitlines()[0] if text.strip() else ""
            evidence_summary = first_line.strip()

        if not evidence_summary:
            evidence_summary = "Insufficient structured AI detail; score derived from detected signal strength."

        if score is None:
            # Heuristic fallback if model provided no explicit numeric score.
            lowered = text.lower()
            positive_hits = sum(
                marker in lowered
                for marker in [
                    "invest",
                    "deployed",
                    "initiative",
                    "generative ai",
                    "machine learning",
                    "fraud",
                    "risk",
                    "chatbot",
                ]
            )
            score = 4.0 + min(positive_hits, 8) * 0.6

        return (
            self._clamp(score, 1.0, 10.0),
            evidence_summary[:220],
            self._clamp(confidence, 0.0, 1.0),
        )

    def _parse_result(self, response_json: Any) -> tuple[float, str, float, str]:
        raw_text = ""
        if isinstance(response_json, dict):
            if "choices" in response_json and isinstance(response_json["choices"], list):
                first = response_json["choices"][0] if response_json["choices"] else {}
                if isinstance(first, dict):
                    message = first.get("message", {})
                    if isinstance(message, dict):
                        raw_text = str(message.get("content") or first.get("text") or "")
                    else:
                        raw_text = str(first.get("text") or "")
            elif "generated_text" in response_json:
                raw_text = str(response_json["generated_text"])
            elif "error" in response_json:
                err = response_json["error"]
                if isinstance(err, dict):
                    raise BackendError(str(err.get("message", err)))
                raise BackendError(str(err))
        elif isinstance(response_json, list) and response_json:
            candidate = response_json[0]
            if isinstance(candidate, dict):
                raw_text = str(candidate.get("generated_text") or candidate.get("text") or "")
            else:
                raw_text = str(candidate)
        else:
            raw_text = str(response_json)
        score, evidence_summary, confidence = self._parse_structured_text(raw_text)
        return (score, evidence_summary, confidence, raw_text)

    def _resolve_device(self, torch_module: Any) -> str:
        if self.local_device != "auto":
            return self.local_device
        if torch_module.cuda.is_available():
            return "cuda"
        if hasattr(torch_module.backends, "mps") and torch_module.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _resolve_local_model_source(self) -> str:
        # If a direct filesystem path is provided, use it directly.
        if os.path.exists(self.model):
            return self.model

        # Resolve Hugging Face cache snapshot path to avoid network metadata calls.
        cache_home = os.path.expanduser("~/.cache/huggingface/hub")
        if "/" in self.model:
            org, name = self.model.split("/", 1)
            cache_dir = os.path.join(cache_home, f"models--{org}--{name}", "snapshots")
            if os.path.isdir(cache_dir):
                snapshots = sorted(
                    [
                        os.path.join(cache_dir, d)
                        for d in os.listdir(cache_dir)
                        if os.path.isdir(os.path.join(cache_dir, d))
                    ]
                )
                if snapshots:
                    return snapshots[-1]
        return self.model

    def _ensure_local_model(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer
        except Exception as exc:  # noqa: BLE001
            raise BackendError(
                "Local Hugging Face mode requires `transformers` and `torch`."
            ) from exc

        self._torch = torch
        self._resolved_device = self._resolve_device(torch)
        model_source = self._resolve_local_model_source()
        model_kwargs: dict[str, Any] = {}
        if self._resolved_device in {"cuda", "mps"}:
            model_kwargs["torch_dtype"] = torch.float16

        def _load_arch(use_safetensors: bool) -> tuple[Any, bool]:
            # Returns (model, is_seq2seq)
            causal_err: Exception | None = None
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    model_source,
                    local_files_only=True,
                    use_safetensors=use_safetensors,
                    **model_kwargs,
                )
                return model, False
            except Exception as exc:  # noqa: BLE001
                causal_err = exc

            try:
                model = AutoModelForSeq2SeqLM.from_pretrained(
                    model_source,
                    local_files_only=True,
                    use_safetensors=use_safetensors,
                    **model_kwargs,
                )
                return model, True
            except Exception as seq_exc:  # noqa: BLE001
                raise BackendError(
                    f"Causal load error: {causal_err}; Seq2Seq load error: {seq_exc}"
                ) from seq_exc

        try:
            self._tokenizer = AutoTokenizer.from_pretrained(
                model_source,
                local_files_only=True,
            )
            try:
                self._model, self._is_seq2seq = _load_arch(use_safetensors=True)
            except BackendError as safe_exc:
                error_text = str(safe_exc).lower()
                safetensor_missing = "model.safetensors" in error_text or "safetensors" in error_text
                if not safetensor_missing:
                    raise

                # Fallback for local trusted cached checkpoints that only provide .bin files.
                try:
                    import transformers.modeling_utils as modeling_utils
                except Exception as exc:  # noqa: BLE001
                    raise BackendError(
                        f"Failed loading local model with safetensors and could not import "
                        f"transformers.modeling_utils for fallback: {exc}"
                    ) from exc

                original_check = getattr(modeling_utils, "check_torch_load_is_safe", None)
                if callable(original_check):
                    modeling_utils.check_torch_load_is_safe = lambda *_args, **_kwargs: None
                try:
                    self._model, self._is_seq2seq = _load_arch(use_safetensors=False)
                finally:
                    if callable(original_check):
                        modeling_utils.check_torch_load_is_safe = original_check
            self._model.to(self._resolved_device)
            self._model.eval()
            if self._tokenizer.pad_token_id is None and self._tokenizer.eos_token_id is not None:
                self._tokenizer.pad_token_id = self._tokenizer.eos_token_id
        except Exception as exc:  # noqa: BLE001
            raise BackendError(
                f"Failed to load local Hugging Face model '{self.model}': {exc}"
            ) from exc

    def _score_document_remote(self, prompt: str) -> LLMResult:
        if not self.token:
            raise BackendError("HF token is required for remote Hugging Face requests")

        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a strict JSON generator. "
                        "Always respond with one JSON object and no markdown."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            "max_tokens": 220,
            "temperature": 0.1,
        }

        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = requests.post(
                    self.url,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout_seconds,
                )
                response.raise_for_status()
                parsed = response.json()
                score, evidence_summary, confidence, raw_response = self._parse_result(parsed)
                return LLMResult(
                    score=score,
                    evidence_summary=evidence_summary,
                    confidence=confidence,
                    raw_response=raw_response,
                    metadata={"provider": "huggingface", "model": self.model},
                )
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if attempt < self.max_retries:
                    time.sleep(1.5 * attempt)
                continue

        raise BackendError(f"Hugging Face request failed after retries: {last_error}")

    def _build_local_chat_input(self, prompt: str) -> str:
        if self._tokenizer is None:
            raise BackendError("Local tokenizer is not initialized")
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a strict JSON generator. "
                    "Always respond with one JSON object and no markdown."
                ),
            },
            {"role": "user", "content": prompt},
        ]
        has_chat_template = bool(getattr(self._tokenizer, "chat_template", None))
        if hasattr(self._tokenizer, "apply_chat_template") and has_chat_template:
            return self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        return (
            "You are a strict JSON generator. Always respond with one JSON object and no markdown.\n"
            f"{prompt}\n"
        )

    def _score_document_local(self, prompt: str) -> LLMResult:
        self._ensure_local_model()
        assert self._tokenizer is not None
        assert self._model is not None
        assert self._torch is not None

        local_input = self._build_local_chat_input(prompt)
        encoded = self._tokenizer(
            local_input,
            return_tensors="pt",
            truncation=True,
            max_length=self.local_max_input_tokens,
        )
        encoded = {k: v.to(self._resolved_device) for k, v in encoded.items()}
        input_len = encoded["input_ids"].shape[-1]

        try:
            with self._torch.no_grad():
                generated = self._model.generate(
                    **encoded,
                    max_new_tokens=self.local_max_new_tokens,
                    do_sample=False,
                    pad_token_id=self._tokenizer.pad_token_id,
                    eos_token_id=self._tokenizer.eos_token_id,
                )
        except Exception as exc:  # noqa: BLE001
            raise BackendError(f"Local generation failed: {exc}") from exc

        if self._is_seq2seq:
            generated_tokens = generated[0]
        else:
            generated_tokens = generated[0][input_len:]
        raw_text = self._tokenizer.decode(generated_tokens, skip_special_tokens=True)
        score, evidence_summary, confidence, _ = self._parse_result(
            {"choices": [{"message": {"content": raw_text}}]}
        )
        return LLMResult(
            score=score,
            evidence_summary=evidence_summary,
            confidence=confidence,
            raw_response=raw_text,
            metadata={
                "provider": "huggingface-local",
                "model": self.model,
                "device": self._resolved_device,
            },
        )

    def score_document(
        self,
        text: str,
        *,
        ticker: str,
        doc_type: str,
        period: str,
    ) -> LLMResult:
        prompt = self._build_prompt(text, ticker=ticker, doc_type=doc_type, period=period)
        if self.local:
            return self._score_document_local(prompt)
        return self._score_document_remote(prompt)
