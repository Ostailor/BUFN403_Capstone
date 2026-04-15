from __future__ import annotations

import json
import logging
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

_CONFIDENCE_LABELS = {
    "very high": 0.95,
    "high": 0.85,
    "medium": 0.60,
    "moderate": 0.60,
    "low": 0.35,
    "very low": 0.15,
}

log = logging.getLogger(__name__)


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


def normalize_confidence(raw_value: Any) -> float:
    if isinstance(raw_value, (int, float)):
        confidence = float(raw_value)
    elif isinstance(raw_value, str):
        value = raw_value.strip().lower()
        if not value:
            confidence = 0.0
        elif value in _CONFIDENCE_LABELS:
            confidence = _CONFIDENCE_LABELS[value]
        else:
            if value.endswith("%"):
                value = value[:-1].strip()
            confidence = float(value)
            if confidence > 1.0:
                confidence /= 100.0
    else:
        confidence = 0.0
    return max(0.0, min(confidence, 1.0))


def normalize_classification_record(
    record: dict[str, Any],
    *,
    fallback_snippet: str = "",
) -> dict[str, Any]:
    intent_level = int(record.get("intent_level", 1))
    if intent_level not in INTENT_LEVELS:
        intent_level = 1

    confidence = normalize_confidence(record.get("confidence", 0.0))

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


def read_classifications_jsonl(path: Path, *, drop_malformed_tail: bool = False) -> list[dict[str, Any]]:
    if not path.exists():
        return []

    records: list[dict[str, Any]] = []
    lines = path.read_bytes().splitlines()
    last_non_empty_index = max((index for index, line in enumerate(lines) if line.strip()), default=-1)
    dropped_tail = False
    for index, raw_line in enumerate(lines):
        if not raw_line.strip():
            continue
        try:
            payload = raw_line.decode("utf-8").strip()
        except UnicodeDecodeError:
            is_malformed_tail = drop_malformed_tail and index == last_non_empty_index
            if is_malformed_tail:
                log.warning("Dropping malformed trailing classification record from %s", path)
                dropped_tail = True
                break
            raise
        if not payload:
            continue
        try:
            raw_record = json.loads(payload)
        except json.JSONDecodeError:
            is_malformed_tail = drop_malformed_tail and index == last_non_empty_index
            if is_malformed_tail:
                log.warning("Dropping malformed trailing classification record from %s", path)
                dropped_tail = True
                break
            raise
        records.append(
            normalize_classification_record(
                raw_record,
                fallback_snippet=str(raw_record.get("chunk_text", "")),
            )
        )
    if dropped_tail:
        write_classifications_jsonl(path, records)
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


def append_classification_record(path: Path, record: dict[str, Any]) -> dict[str, Any]:
    normalized = normalize_classification_record(
        dict(record),
        fallback_snippet=str(record.get("chunk_text", "")),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(normalized) + "\n")
        handle.flush()
    return normalized
