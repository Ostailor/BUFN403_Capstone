from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from .config import INTENT_LEVELS

CLASSIFICATION_COLUMNS = [
    "chunk_id",
    "ticker",
    "bank_name",
    "source_type",
    "period_year",
    "period_quarter",
    "intent_level",
    "intent_label",
    "app_categories",
    "confidence",
    "evidence_snippet",
]


def normalize_app_categories(raw_value: Any) -> list[str]:
    if isinstance(raw_value, list):
        return [str(value).strip() for value in raw_value if str(value).strip()]
    if isinstance(raw_value, str):
        value = raw_value.strip()
        if not value:
            return []
        if value.startswith("["):
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, list):
                return [str(item).strip() for item in parsed if str(item).strip()]
        return [part.strip() for part in value.split(",") if part.strip()]
    return []


def normalize_classification_record(
    record: dict[str, Any],
    *,
    fallback_snippet: str = "",
) -> dict[str, Any]:
    intent_level = int(record.get("intent_level", 1))
    if intent_level not in INTENT_LEVELS:
        intent_level = 1

    confidence = float(record.get("confidence", 0.0))
    confidence = max(0.0, min(confidence, 1.0))

    evidence_snippet = str(record.get("evidence_snippet", "")).strip() or fallback_snippet[:300]

    return {
        "chunk_id": str(record.get("chunk_id", "")).strip(),
        "ticker": str(record.get("ticker", "")).strip(),
        "bank_name": str(record.get("bank_name", "")).strip(),
        "source_type": str(record.get("source_type", "")).strip(),
        "period_year": int(record.get("period_year", 0) or 0),
        "period_quarter": int(record.get("period_quarter", 0) or 0),
        "intent_level": intent_level,
        "intent_label": INTENT_LEVELS[intent_level],
        "app_categories": normalize_app_categories(record.get("app_categories", [])),
        "confidence": confidence,
        "evidence_snippet": evidence_snippet,
    }


def read_classifications_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []

    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        payload = line.strip()
        if not payload:
            continue
        raw_record = json.loads(payload)
        records.append(
            normalize_classification_record(
                raw_record,
                fallback_snippet=str(raw_record.get("chunk_text", "")),
            )
        )
    return records


def write_classifications_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            normalized = normalize_classification_record(
                dict(record),
                fallback_snippet=str(record.get("chunk_text", "")),
            )
            handle.write(json.dumps(normalized) + "\n")
    return path
