from __future__ import annotations

import json
import logging
import time
from typing import Any

from .classification_io import (
    append_classification_record,
    normalize_app_categories,
    normalize_classification_record,
    read_classifications_jsonl,
    write_classifications_jsonl,
)
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


def _write_progress(
    paths: CorpusPaths,
    *,
    status: str,
    total_chunks: int,
    completed_chunks: int,
    resumed_chunks: int,
    processed_this_run: int,
    last_chunk_id: str = "",
    failed_chunk_id: str = "",
    elapsed_seconds: float = 0.0,
) -> None:
    session_throughput = processed_this_run / elapsed_seconds if elapsed_seconds > 0 else 0.0
    remaining_chunks = max(total_chunks - completed_chunks, 0)
    eta_seconds = remaining_chunks / session_throughput if session_throughput > 0 else None
    payload = {
        "status": status,
        "total_chunks": total_chunks,
        "completed_chunks": completed_chunks,
        "resumed_chunks": resumed_chunks,
        "processed_this_run": processed_this_run,
        "pending_chunks": remaining_chunks,
        "last_chunk_id": last_chunk_id,
        "failed_chunk_id": failed_chunk_id,
        "elapsed_seconds": round(elapsed_seconds, 2),
        "session_elapsed_seconds": round(elapsed_seconds, 2),
        "throughput_chunks_per_second": round(session_throughput, 4),
        "session_throughput_chunks_per_second": round(session_throughput, 4),
        "eta_seconds": round(eta_seconds, 2) if eta_seconds is not None else None,
        "session_eta_seconds": round(eta_seconds, 2) if eta_seconds is not None else None,
    }
    paths.classification_progress_json.parent.mkdir(parents=True, exist_ok=True)
    paths.classification_progress_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _log_progress(
    *,
    completed_chunks: int,
    total_chunks: int,
    resumed_chunks: int,
    processed_this_run: int,
    elapsed_seconds: float,
    chunk_id: str,
) -> None:
    throughput = processed_this_run / elapsed_seconds if elapsed_seconds > 0 else 0.0
    remaining_chunks = max(total_chunks - completed_chunks, 0)
    eta_seconds = remaining_chunks / throughput if throughput > 0 else 0.0
    log.info(
        "Classification progress %d/%d complete (resumed=%d, this-run=%d, current=%s, session-elapsed=%.1fs, session-rate=%.2f chunks/s, session-eta=%.1fs)",
        completed_chunks,
        total_chunks,
        resumed_chunks,
        processed_this_run,
        chunk_id,
        elapsed_seconds,
        throughput,
        eta_seconds,
    )


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
    resume: bool = True,
    log_every: int | None = None,
) -> Path:
    chunks_path = paths.chunks_jsonl
    if not chunks_path.exists():
        raise FileNotFoundError(f"Chunks file not found: {chunks_path}")

    with open(chunks_path) as f:
        all_chunks = [json.loads(line) for line in f if line.strip()]

    ai_chunks = [c for c in all_chunks if has_ai_anchor(c.get("chunk_text", ""))]
    log.info("Found %d AI-anchor chunks out of %d total", len(ai_chunks), len(all_chunks))

    output_path = paths.classifications_jsonl
    existing_records = read_classifications_jsonl(output_path, drop_malformed_tail=True) if resume else []
    completed_chunk_ids = {
        record["chunk_id"]
        for record in existing_records
        if record.get("chunk_id")
    }
    resumed_chunks = len(completed_chunk_ids)
    pending_chunks = [chunk for chunk in ai_chunks if chunk["chunk_id"] not in completed_chunk_ids]

    if not resume:
        write_classifications_jsonl(output_path, [])
    elif output_path.exists():
        write_classifications_jsonl(output_path, existing_records)

    log.info(
        "Starting classification run with %d pending chunks (%d already complete, resume=%s)",
        len(pending_chunks),
        resumed_chunks,
        resume,
    )

    if not pending_chunks:
        _write_progress(
            paths,
            status="complete",
            total_chunks=len(ai_chunks),
            completed_chunks=resumed_chunks,
            resumed_chunks=resumed_chunks,
            processed_this_run=0,
            elapsed_seconds=0.0,
        )
        log.info("No pending chunks remain; classification output is already complete at %s", output_path)
        return output_path

    qwen = QwenAnswerGenerator(prefer_local=prefer_local, local_files_only=local_files_only)
    started_at = time.monotonic()
    log_interval = max(1, log_every or batch_size)

    _write_progress(
        paths,
        status="running",
        total_chunks=len(ai_chunks),
        completed_chunks=resumed_chunks,
        resumed_chunks=resumed_chunks,
        processed_this_run=0,
        elapsed_seconds=0.0,
    )

    try:
        for i, chunk in enumerate(pending_chunks, start=1):
            result = classify_chunk(chunk, qwen)
            normalized = append_classification_record(output_path, result)
            completed_chunks = resumed_chunks + i
            elapsed_seconds = time.monotonic() - started_at
            chunk_id = normalized.get("chunk_id", chunk.get("chunk_id", "unknown"))
            _write_progress(
                paths,
                status="running",
                total_chunks=len(ai_chunks),
                completed_chunks=completed_chunks,
                resumed_chunks=resumed_chunks,
                processed_this_run=i,
                last_chunk_id=chunk_id,
                elapsed_seconds=elapsed_seconds,
            )
            if completed_chunks == len(ai_chunks) or i == 1 or i % log_interval == 0:
                _log_progress(
                    completed_chunks=completed_chunks,
                    total_chunks=len(ai_chunks),
                    resumed_chunks=resumed_chunks,
                    processed_this_run=i,
                    elapsed_seconds=elapsed_seconds,
                    chunk_id=chunk_id,
                )
    except Exception:
        completed_now = len(read_classifications_jsonl(output_path, drop_malformed_tail=True))
        elapsed_seconds = time.monotonic() - started_at
        failed_chunk_id = chunk.get("chunk_id", "unknown")
        processed_this_run = max(completed_now - resumed_chunks, 0)
        _write_progress(
            paths,
            status="failed",
            total_chunks=len(ai_chunks),
            completed_chunks=completed_now,
            resumed_chunks=resumed_chunks,
            processed_this_run=processed_this_run,
            failed_chunk_id=failed_chunk_id,
            elapsed_seconds=elapsed_seconds,
        )
        log.exception("Classification run failed at chunk %s after %d completed chunks", failed_chunk_id, completed_now)
        raise

    total_completed = len(read_classifications_jsonl(output_path, drop_malformed_tail=True))
    elapsed_seconds = time.monotonic() - started_at
    _write_progress(
        paths,
        status="complete",
        total_chunks=len(ai_chunks),
        completed_chunks=total_completed,
        resumed_chunks=resumed_chunks,
        processed_this_run=max(total_completed - resumed_chunks, 0),
        elapsed_seconds=elapsed_seconds,
    )
    log.info("Wrote %d classifications to %s", total_completed, output_path)
    return output_path
