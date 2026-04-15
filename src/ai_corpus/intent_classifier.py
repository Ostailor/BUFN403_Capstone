from __future__ import annotations

import json
import logging
from typing import Any

from .classification_io import normalize_app_categories, normalize_classification_record, write_classifications_jsonl
from .config import APP_CATEGORIES, CLASSIFIER_BATCH_SIZE, INTENT_LEVELS, INTENT_SIGNAL_WORDS, CorpusPaths
from .pipeline import has_ai_anchor
from .qwen import QwenAnswerGenerator

log = logging.getLogger(__name__)

_VALID_APP_CATEGORIES = set(APP_CATEGORIES.keys())
CLASSIFICATION_SYSTEM_PROMPT = (
    "You classify bank AI mentions by intent level and application type. "
    "Return exactly one JSON object and no markdown."
)


class ClassificationError(RuntimeError):
    pass


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

    return normalize_classification_record(
        {
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
        },
        fallback_snippet=chunk["chunk_text"],
    )


def build_classification_messages(chunk: dict[str, Any]) -> list[dict[str, str]]:
    ticker = chunk.get("ticker", "unknown")
    source_type = chunk.get("source_type", "filing")
    chunk_text = chunk["chunk_text"]

    user_msg = (
        f'Classify this text from {ticker}\'s {source_type}:\n\n'
        f'"{chunk_text}"\n\n'
        "Intent levels: 1=Exploring (investigating/piloting), 2=Committing (investing/building), "
        "3=Deploying (launched/operational), 4=Scaling (enterprise-wide/expanding)\n\n"
        "Application categories (pick all that apply): GenAI / LLMs, Predictive ML, "
        "NLP / Text, Computer Vision, RPA / Automation, Fraud / Risk Models\n\n"
        "Return JSON with keys: intent_level, intent_label, app_categories, confidence, evidence_snippet."
    )
    return [
        {"role": "system", "content": CLASSIFICATION_SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]


def _normalize_llm_payload(chunk: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    raw_cats = [
        category
        for category in normalize_app_categories(payload.get("app_categories", []))
        if category in _VALID_APP_CATEGORIES
    ]
    return normalize_classification_record(
        {
            "chunk_id": chunk["chunk_id"],
            "ticker": chunk.get("ticker"),
            "bank_name": chunk.get("bank_name"),
            "source_type": chunk.get("source_type"),
            "period_year": chunk.get("period_year"),
            "period_quarter": chunk.get("period_quarter"),
            "intent_level": payload.get("intent_level", 1),
            "intent_label": payload.get("intent_label", ""),
            "app_categories": raw_cats,
            "confidence": payload.get("confidence", 0.5),
            "evidence_snippet": payload.get("evidence_snippet", chunk["chunk_text"][:200]),
        },
        fallback_snippet=chunk["chunk_text"],
    )


def classify_chunk(chunk: dict, qwen: QwenAnswerGenerator) -> dict:
    try:
        payload = qwen.generate_json(messages=build_classification_messages(chunk))
        return _normalize_llm_payload(chunk, payload)
    except Exception as exc:
        chunk_id = chunk.get("chunk_id", "unknown")
        raise ClassificationError(f"Qwen classification failed for chunk {chunk_id}") from exc


def run_classification(
    paths: CorpusPaths,
    batch_size: int = CLASSIFIER_BATCH_SIZE,
    *,
    prefer_local: bool = True,
    local_files_only: bool = True,
) -> Path:
    chunks_path = paths.chunks_jsonl
    if not chunks_path.exists():
        raise FileNotFoundError(f"Chunks file not found: {chunks_path}")

    with open(chunks_path) as f:
        all_chunks = [json.loads(line) for line in f if line.strip()]

    ai_chunks = [c for c in all_chunks if has_ai_anchor(c.get("chunk_text", ""))]
    log.info("Found %d AI-anchor chunks out of %d total", len(ai_chunks), len(all_chunks))

    qwen = QwenAnswerGenerator(prefer_local=prefer_local, local_files_only=local_files_only)
    results: list[dict] = []

    for i, chunk in enumerate(ai_chunks):
        result = classify_chunk(chunk, qwen)
        results.append(result)
        if (i + 1) % batch_size == 0:
            log.info("Classified %d / %d chunks", i + 1, len(ai_chunks))

    output_path = paths.classifications_jsonl
    write_classifications_jsonl(output_path, results)

    log.info("Wrote %d classifications to %s", len(results), output_path)
    return output_path
