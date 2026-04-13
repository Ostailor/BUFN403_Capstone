from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

import requests

from .config import (
    APP_CATEGORIES,
    CLASSIFIER_BATCH_SIZE,
    INTENT_LEVELS,
    INTENT_SIGNAL_WORDS,
    CorpusPaths,
)
from .pipeline import has_ai_anchor
from .qwen import QwenAnswerGenerator

log = logging.getLogger(__name__)

_VALID_APP_CATEGORIES = set(APP_CATEGORIES.keys())


def classify_chunk_rules(chunk: dict) -> dict:
    text_lower = chunk["chunk_text"].lower()

    intent_level = 1
    for level in (4, 3, 2, 1):
        if any(word in text_lower for word in INTENT_SIGNAL_WORDS[level]):
            intent_level = level
            break

    matched_categories: list[str] = []
    for category, keywords in APP_CATEGORIES.items():
        if any(kw in text_lower for kw in keywords):
            matched_categories.append(category)

    return {
        "chunk_id": chunk["chunk_id"],
        "ticker": chunk.get("ticker"),
        "bank_name": chunk.get("bank_name"),
        "source_type": chunk.get("source_type"),
        "period_year": chunk.get("period_year"),
        "period_quarter": chunk.get("period_quarter"),
        "intent_level": intent_level,
        "intent_label": INTENT_LEVELS[intent_level],
        "app_categories": matched_categories,
        "confidence": 0.5,
        "evidence_snippet": chunk["chunk_text"][:200],
    }


def classify_chunk(chunk: dict, qwen: QwenAnswerGenerator) -> dict:
    ticker = chunk.get("ticker", "unknown")
    source_type = chunk.get("source_type", "filing")
    chunk_text = chunk["chunk_text"]

    system_msg = "You classify bank AI mentions by intent level and application type. Return JSON only."
    user_msg = (
        f'Classify this text from {ticker}\'s {source_type}:\n\n'
        f'"{chunk_text}"\n\n'
        f'Intent levels: 1=Exploring (investigating/piloting), 2=Committing (investing/building), '
        f'3=Deploying (launched/operational), 4=Scaling (enterprise-wide/expanding)\n\n'
        f'Application categories (pick all that apply): GenAI / LLMs, Predictive ML, '
        f'NLP / Text, Computer Vision, RPA / Automation, Fraud / Risk Models\n\n'
        f'Return JSON: {{intent_level: int, intent_label: str, app_categories: [str], '
        f'confidence: float, evidence_snippet: str}}'
    )

    try:
        resp = requests.post(
            "https://router.huggingface.co/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {qwen.hf_token}",
                "Content-Type": "application/json",
            },
            json={
                "model": qwen.model_candidates[0],
                "messages": [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
                "temperature": 0.1,
                "max_tokens": 500,
            },
            timeout=90,
        )
        resp.raise_for_status()
        raw_text = resp.json()["choices"][0]["message"]["content"]
        match = re.search(r"\{.*\}", raw_text, flags=re.S)
        if not match:
            raise ValueError("No JSON found in response")
        payload: dict[str, Any] = json.loads(match.group(0))

        intent_level = int(payload.get("intent_level", 1))
        if intent_level not in INTENT_LEVELS:
            intent_level = 1

        raw_cats = payload.get("app_categories", [])
        app_categories = [c for c in raw_cats if c in _VALID_APP_CATEGORIES]

        return {
            "chunk_id": chunk["chunk_id"],
            "ticker": chunk.get("ticker"),
            "bank_name": chunk.get("bank_name"),
            "source_type": chunk.get("source_type"),
            "period_year": chunk.get("period_year"),
            "period_quarter": chunk.get("period_quarter"),
            "intent_level": intent_level,
            "intent_label": INTENT_LEVELS[intent_level],
            "app_categories": app_categories,
            "confidence": float(payload.get("confidence", 0.5)),
            "evidence_snippet": str(payload.get("evidence_snippet", chunk_text[:200])),
        }
    except Exception:
        log.warning("Qwen classification failed for chunk %s, falling back to rules", chunk.get("chunk_id"))
        return classify_chunk_rules(chunk)


def run_classification(paths: CorpusPaths, batch_size: int = CLASSIFIER_BATCH_SIZE) -> Path:
    chunks_path = paths.chunks_jsonl
    if not chunks_path.exists():
        raise FileNotFoundError(f"Chunks file not found: {chunks_path}")

    with open(chunks_path) as f:
        all_chunks = [json.loads(line) for line in f if line.strip()]

    ai_chunks = [c for c in all_chunks if has_ai_anchor(c.get("chunk_text", ""))]
    log.info("Found %d AI-anchor chunks out of %d total", len(ai_chunks), len(all_chunks))

    qwen = QwenAnswerGenerator(prefer_local=True)
    results: list[dict] = []

    for i, chunk in enumerate(ai_chunks):
        result = classify_chunk(chunk, qwen)
        results.append(result)
        if (i + 1) % batch_size == 0:
            log.info("Classified %d / %d chunks", i + 1, len(ai_chunks))

    output_path = paths.classifications_jsonl
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    log.info("Wrote %d classifications to %s", len(results), output_path)
    return output_path
