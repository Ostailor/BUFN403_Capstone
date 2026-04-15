from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ai_corpus.intent_classifier import (
    ClassificationError,
    build_classification_messages,
    classify_chunk,
    classify_chunk_rules,
    run_classification,
)
from src.ai_corpus.classification_io import normalize_confidence, read_classifications_jsonl
from src.ai_corpus.qwen import QwenAnswerGenerator


def test_classify_chunk_rules_detects_deploying_intent() -> None:
    chunk = {
        "chunk_id": "AAA_2024_Q1_001",
        "ticker": "AAA",
        "bank_name": "Alpha Bank",
        "source_type": "transcript",
        "period_year": 2024,
        "period_quarter": 1,
        "chunk_text": "We have launched our generative AI chatbot across retail banking",
    }
    result = classify_chunk_rules(chunk)
    assert result["intent_level"] == 3, f"Expected Deploying (3), got {result['intent_level']}"
    assert "GenAI / LLMs" in result["app_categories"]
    assert result["confidence"] == 0.5


def test_classify_chunk_rules_detects_exploring_intent() -> None:
    chunk = {
        "chunk_id": "BBB_2024_Q2_010",
        "ticker": "BBB",
        "bank_name": "Beta Bank",
        "source_type": "10-K",
        "period_year": 2024,
        "period_quarter": 2,
        "chunk_text": "We are currently evaluating machine learning solutions for credit scoring",
    }
    result = classify_chunk_rules(chunk)
    assert result["intent_level"] == 1, f"Expected Exploring (1), got {result['intent_level']}"
    assert "Predictive ML" in result["app_categories"]


def test_classify_chunk_rules_detects_scaling_intent() -> None:
    chunk = {
        "chunk_id": "CCC_2024_Q3_005",
        "ticker": "CCC",
        "bank_name": "Gamma Bank",
        "source_type": "transcript",
        "period_year": 2024,
        "period_quarter": 3,
        "chunk_text": "We are expanding our AI-powered fraud detection enterprise-wide",
    }
    result = classify_chunk_rules(chunk)
    assert result["intent_level"] == 4, f"Expected Scaling (4), got {result['intent_level']}"
    assert "Fraud / Risk Models" in result["app_categories"]


def test_classify_chunk_rules_detects_multiple_categories() -> None:
    chunk = {
        "chunk_id": "AAA_2024_Q4_020",
        "ticker": "AAA",
        "bank_name": "Alpha Bank",
        "source_type": "10-Q",
        "period_year": 2024,
        "period_quarter": 4,
        "chunk_text": "Our chatbot uses NLP for document processing and we deployed fraud detection models",
    }
    result = classify_chunk_rules(chunk)
    assert len(result["app_categories"]) >= 2
    assert "GenAI / LLMs" in result["app_categories"]
    assert "Fraud / Risk Models" in result["app_categories"]


def test_classify_chunk_rules_defaults_to_exploring() -> None:
    chunk = {
        "chunk_id": "BBB_2024_Q1_099",
        "ticker": "BBB",
        "bank_name": "Beta Bank",
        "source_type": "transcript",
        "period_year": 2024,
        "period_quarter": 1,
        "chunk_text": "artificial intelligence is mentioned here",
    }
    result = classify_chunk_rules(chunk)
    assert result["intent_level"] == 1, "Should default to Exploring (1) with no signal words"


def test_classify_chunk_returns_required_fields() -> None:
    chunk = {
        "chunk_id": "AAA_2024_Q1_001",
        "ticker": "AAA",
        "bank_name": "Alpha Bank",
        "source_type": "transcript",
        "period_year": 2024,
        "period_quarter": 1,
        "chunk_text": "We launched an AI chatbot for customer service",
    }
    result = classify_chunk_rules(chunk)
    required_keys = {
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
    }
    assert required_keys.issubset(result.keys()), (
        f"Missing keys: {required_keys - result.keys()}"
    )


def test_classify_chunk_raises_on_qwen_failure() -> None:
    chunk = {
        "chunk_id": "AAA_2024_Q2_007",
        "ticker": "AAA",
        "bank_name": "Alpha Bank",
        "source_type": "10-K",
        "period_year": 2024,
        "period_quarter": 2,
        "chunk_text": "We have launched our generative AI chatbot across retail banking",
    }
    mock_qwen = MagicMock()
    mock_qwen.generate_json.side_effect = RuntimeError("API unavailable")
    try:
        classify_chunk(chunk, mock_qwen)
    except ClassificationError as exc:
        assert "AAA_2024_Q2_007" in str(exc)
    else:
        raise AssertionError("classify_chunk should raise when Qwen classification fails")


def test_classify_chunk_uses_qwen_payload_when_available() -> None:
    chunk = {
        "chunk_id": "AAA_2024_Q3_011",
        "ticker": "AAA",
        "bank_name": "Alpha Bank",
        "source_type": "transcript",
        "period_year": 2024,
        "period_quarter": 3,
        "chunk_text": "We deployed copilots for fraud operations and client servicing.",
    }
    mock_qwen = MagicMock()
    mock_qwen.generate_json.return_value = {
        "intent_level": 4,
        "intent_label": "Scaling",
        "app_categories": ["GenAI / LLMs", "Fraud / Risk Models", "Invalid Category"],
        "confidence": 0.92,
        "evidence_snippet": "We deployed copilots for fraud operations.",
    }

    result = classify_chunk(chunk, mock_qwen)

    assert result["intent_level"] == 4
    assert result["intent_label"] == "Scaling"
    assert result["app_categories"] == ["GenAI / LLMs", "Fraud / Risk Models"]
    assert result["confidence"] == 0.92
    mock_qwen.generate_json.assert_called_once_with(messages=build_classification_messages(chunk))


def test_normalize_confidence_accepts_label_values() -> None:
    assert normalize_confidence("High") == 0.85
    assert normalize_confidence("medium") == 0.60
    assert normalize_confidence("25%") == 0.25


def test_classify_chunk_accepts_string_confidence_from_qwen() -> None:
    chunk = {
        "chunk_id": "MS_transcript_2025_Q3__sec_003__chunk_0003",
        "ticker": "MS",
        "bank_name": "Morgan Stanley",
        "source_type": "transcript",
        "period_year": 2025,
        "period_quarter": 3,
        "chunk_text": "We are deploying AI assistants across operations.",
    }
    mock_qwen = MagicMock()
    mock_qwen.generate_json.return_value = {
        "intent_level": 3,
        "intent_label": "Deploying",
        "app_categories": ["RPA / Automation"],
        "confidence": "High",
        "evidence_snippet": "deploying AI assistants across operations",
    }

    result = classify_chunk(chunk, mock_qwen)

    assert result["confidence"] == 0.85


def test_qwen_extract_json_recovers_truncated_but_structured_output() -> None:
    generator = QwenAnswerGenerator()
    raw_text = """{
  "intent_level": 3,
  "intent_label": "Deploying",
  "app_categories": ["GenAI / LLMs"],
  "confidence": 0.9,
  "evidence_snippet": "We're excited and incredibly grateful for the trust"""

    payload = generator._extract_json(raw_text)

    assert payload["intent_level"] == 3
    assert payload["intent_label"] == "Deploying"
    assert payload["app_categories"] == ["GenAI / LLMs"]
    assert payload["confidence"] == "0.9" or payload["confidence"] == 0.9
    assert payload["evidence_snippet"].startswith("We're excited")


def test_qwen_extract_json_recovers_malformed_full_json() -> None:
    generator = QwenAnswerGenerator()
    raw_text = """{
  "intent_level": 3,
  "intent_label": "Deploying",
  "app_categories": ["GenAI / LLMs"],
  "confidence": 0.9,
  "evidence_snippet": "We are expanding copilots "across operations""
}"""

    payload = generator._extract_json(raw_text)

    assert payload["intent_level"] == 3
    assert payload["intent_label"] == "Deploying"
    assert payload["app_categories"] == ["GenAI / LLMs"]
    assert payload["evidence_snippet"].startswith("We are expanding copilots")


def test_read_classifications_jsonl_drops_malformed_trailing_record(tmp_path: Path) -> None:
    path = tmp_path / "classifications.jsonl"
    path.write_text(
        json.dumps(
            {
                "chunk_id": "AAA_001",
                "ticker": "AAA",
                "bank_name": "Alpha Bank",
                "source_type": "transcript",
                "period_year": 2024,
                "period_quarter": 1,
                "intent_level": 3,
                "intent_label": "Deploying",
                "app_categories": ["GenAI / LLMs"],
                "confidence": 0.8,
                "evidence_snippet": "good",
            }
        )
        + "\n"
        + '{"chunk_id":"AAA_002","intent_level":3'
        + "\n",
        encoding="utf-8",
    )

    records = read_classifications_jsonl(path, drop_malformed_tail=True)

    assert len(records) == 1
    assert records[0]["chunk_id"] == "AAA_001"
    assert path.read_text(encoding="utf-8") == json.dumps(records[0]) + "\n"


def test_read_classifications_jsonl_drops_malformed_trailing_utf8_record(tmp_path: Path) -> None:
    path = tmp_path / "classifications.jsonl"
    good_record = {
        "chunk_id": "AAA_001",
        "ticker": "AAA",
        "bank_name": "Alpha Bank",
        "source_type": "transcript",
        "period_year": 2024,
        "period_quarter": 1,
        "intent_level": 3,
        "intent_label": "Deploying",
        "app_categories": ["GenAI / LLMs"],
        "confidence": 0.8,
        "evidence_snippet": "good",
    }
    path.write_bytes(
        json.dumps(good_record).encode("utf-8")
        + b"\n"
        + b'{"chunk_id":"AAA_002","evidence_snippet":"bad \xe2\x82',
    )

    records = read_classifications_jsonl(path, drop_malformed_tail=True)

    assert len(records) == 1
    assert records[0]["chunk_id"] == "AAA_001"
    assert path.read_text(encoding="utf-8") == json.dumps(records[0]) + "\n"


def test_run_classification_writes_jsonl(tmp_path: Path) -> None:
    chunks_file = tmp_path / "chunks.jsonl"
    chunks = [
        {
            "chunk_id": "AAA_2024_Q1_001",
            "ticker": "AAA",
            "bank_name": "Alpha Bank",
            "source_type": "transcript",
            "period_year": 2024,
            "period_quarter": 1,
            "chunk_text": "We launched an AI chatbot",
            "ai_anchor": True,
        },
        {
            "chunk_id": "AAA_2024_Q1_002",
            "ticker": "AAA",
            "bank_name": "Alpha Bank",
            "source_type": "transcript",
            "period_year": 2024,
            "period_quarter": 1,
            "chunk_text": "Machine learning for fraud detection",
            "ai_anchor": True,
        },
        {
            "chunk_id": "BBB_2024_Q1_001",
            "ticker": "BBB",
            "bank_name": "Beta Bank",
            "source_type": "transcript",
            "period_year": 2024,
            "period_quarter": 1,
            "chunk_text": "We discussed deposits and balance sheet positioning.",
            "ai_anchor": False,
        },
    ]
    with chunks_file.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c) + "\n")

    fixed_result = {
        "chunk_id": "test",
        "ticker": "AAA",
        "bank_name": "Alpha Bank",
        "source_type": "transcript",
        "period_year": 2024,
        "period_quarter": 1,
        "intent_level": 3,
        "intent_label": "Deploying",
        "app_categories": ["GenAI / LLMs"],
        "confidence": 0.85,
        "evidence_snippet": "test snippet",
    }

    output_file = tmp_path / "classifications.jsonl"

    mock_paths = MagicMock()
    mock_paths.chunks_jsonl = chunks_file
    mock_paths.classifications_jsonl = output_file

    with patch(
        "src.ai_corpus.intent_classifier.QwenAnswerGenerator", autospec=True
    ), patch(
        "src.ai_corpus.intent_classifier.classify_chunk", return_value=fixed_result
    ):
        run_classification(mock_paths)

    assert output_file.exists(), "classifications.jsonl should be created"
    lines = output_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2, f"Expected 2 AI-anchor lines, got {len(lines)}"
    for line in lines:
        record = json.loads(line)
        assert "intent_level" in record
        assert "app_categories" in record


def test_run_classification_passes_colab_friendly_flags(tmp_path: Path) -> None:
    chunks_file = tmp_path / "chunks.jsonl"
    chunks_file.write_text(
        json.dumps(
            {
                "chunk_id": "AAA_2024_Q1_001",
                "ticker": "AAA",
                "bank_name": "Alpha Bank",
                "source_type": "transcript",
                "period_year": 2024,
                "period_quarter": 1,
                "chunk_text": "We launched an AI chatbot.",
                "ai_anchor": True,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    mock_paths = MagicMock()
    mock_paths.chunks_jsonl = chunks_file
    mock_paths.classifications_jsonl = tmp_path / "classifications.jsonl"

    with patch(
        "src.ai_corpus.intent_classifier.QwenAnswerGenerator", autospec=True
    ) as generator_cls, patch(
        "src.ai_corpus.intent_classifier.classify_chunk",
        return_value={
            "chunk_id": "AAA_2024_Q1_001",
            "ticker": "AAA",
            "bank_name": "Alpha Bank",
            "source_type": "transcript",
            "period_year": 2024,
            "period_quarter": 1,
            "intent_level": 3,
            "intent_label": "Deploying",
            "app_categories": ["GenAI / LLMs"],
            "confidence": 0.8,
            "evidence_snippet": "We launched an AI chatbot.",
        },
    ):
        run_classification(mock_paths, prefer_local=True, local_files_only=False)

    generator_cls.assert_called_once_with(prefer_local=True, local_files_only=False)


def test_run_classification_resumes_from_existing_jsonl(tmp_path: Path) -> None:
    chunks_file = tmp_path / "chunks.jsonl"
    existing_record = {
        "chunk_id": "AAA_2024_Q1_001",
        "ticker": "AAA",
        "bank_name": "Alpha Bank",
        "source_type": "transcript",
        "period_year": 2024,
        "period_quarter": 1,
        "intent_level": 3,
        "intent_label": "Deploying",
        "app_categories": ["GenAI / LLMs"],
        "confidence": 0.8,
        "evidence_snippet": "existing",
    }
    pending_chunk = {
        "chunk_id": "AAA_2024_Q1_002",
        "ticker": "AAA",
        "bank_name": "Alpha Bank",
        "source_type": "transcript",
        "period_year": 2024,
        "period_quarter": 1,
        "chunk_text": "Machine learning for fraud detection",
        "ai_anchor": True,
    }
    with chunks_file.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps({**existing_record, "chunk_text": "existing", "ai_anchor": True}) + "\n")
        handle.write(json.dumps(pending_chunk) + "\n")

    output_file = tmp_path / "classifications.jsonl"
    output_file.write_text(json.dumps(existing_record) + "\n", encoding="utf-8")
    progress_file = tmp_path / "classification_progress.json"

    mock_paths = MagicMock()
    mock_paths.chunks_jsonl = chunks_file
    mock_paths.classifications_jsonl = output_file
    mock_paths.classification_progress_json = progress_file

    new_record = {
        "chunk_id": "AAA_2024_Q1_002",
        "ticker": "AAA",
        "bank_name": "Alpha Bank",
        "source_type": "transcript",
        "period_year": 2024,
        "period_quarter": 1,
        "intent_level": 2,
        "intent_label": "Committing",
        "app_categories": ["Fraud / Risk Models"],
        "confidence": 0.7,
        "evidence_snippet": "new",
    }

    with patch("src.ai_corpus.intent_classifier.QwenAnswerGenerator", autospec=True), patch(
        "src.ai_corpus.intent_classifier.classify_chunk",
        return_value=new_record,
    ) as classify_mock, patch(
        "src.ai_corpus.intent_classifier.time.monotonic",
        side_effect=[100.0, 104.0, 104.0],
    ):
        run_classification(mock_paths, resume=True, log_every=1)

    classify_mock.assert_called_once()
    lines = output_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    progress = json.loads(progress_file.read_text(encoding="utf-8"))
    assert progress["status"] == "complete"
    assert progress["completed_chunks"] == 2
    assert progress["resumed_chunks"] == 1
    assert progress["processed_this_run"] == 1
    assert progress["throughput_chunks_per_second"] == 0.25


def test_run_classification_persists_progress_before_failure(tmp_path: Path) -> None:
    chunks_file = tmp_path / "chunks.jsonl"
    chunks = [
        {
            "chunk_id": "AAA_2024_Q1_001",
            "ticker": "AAA",
            "bank_name": "Alpha Bank",
            "source_type": "transcript",
            "period_year": 2024,
            "period_quarter": 1,
            "chunk_text": "We launched an AI chatbot",
            "ai_anchor": True,
        },
        {
            "chunk_id": "AAA_2024_Q1_002",
            "ticker": "AAA",
            "bank_name": "Alpha Bank",
            "source_type": "transcript",
            "period_year": 2024,
            "period_quarter": 1,
            "chunk_text": "Machine learning for fraud detection",
            "ai_anchor": True,
        },
    ]
    with chunks_file.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(json.dumps(chunk) + "\n")

    output_file = tmp_path / "classifications.jsonl"
    progress_file = tmp_path / "classification_progress.json"
    mock_paths = MagicMock()
    mock_paths.chunks_jsonl = chunks_file
    mock_paths.classifications_jsonl = output_file
    mock_paths.classification_progress_json = progress_file

    first_record = {
        "chunk_id": "AAA_2024_Q1_001",
        "ticker": "AAA",
        "bank_name": "Alpha Bank",
        "source_type": "transcript",
        "period_year": 2024,
        "period_quarter": 1,
        "intent_level": 3,
        "intent_label": "Deploying",
        "app_categories": ["GenAI / LLMs"],
        "confidence": 0.8,
        "evidence_snippet": "first",
    }

    with patch("src.ai_corpus.intent_classifier.QwenAnswerGenerator", autospec=True), patch(
        "src.ai_corpus.intent_classifier.classify_chunk",
        side_effect=[first_record, ClassificationError("boom")],
    ):
        try:
            run_classification(mock_paths, resume=True, log_every=1)
        except ClassificationError:
            pass
        else:
            raise AssertionError("run_classification should propagate the classification failure")

    lines = output_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    progress = json.loads(progress_file.read_text(encoding="utf-8"))
    assert progress["status"] == "failed"
    assert progress["completed_chunks"] == 1
    assert progress["failed_chunk_id"] == "AAA_2024_Q1_002"


def test_run_classification_repairs_malformed_tail_before_resume(tmp_path: Path) -> None:
    chunks_file = tmp_path / "chunks.jsonl"
    existing_chunk = {
        "chunk_id": "AAA_2024_Q1_001",
        "ticker": "AAA",
        "bank_name": "Alpha Bank",
        "source_type": "transcript",
        "period_year": 2024,
        "period_quarter": 1,
        "chunk_text": "We launched an AI chatbot",
        "ai_anchor": True,
    }
    pending_chunk = {
        "chunk_id": "AAA_2024_Q1_002",
        "ticker": "AAA",
        "bank_name": "Alpha Bank",
        "source_type": "transcript",
        "period_year": 2024,
        "period_quarter": 1,
        "chunk_text": "Machine learning for fraud detection",
        "ai_anchor": True,
    }
    with chunks_file.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(existing_chunk) + "\n")
        handle.write(json.dumps(pending_chunk) + "\n")

    output_file = tmp_path / "classifications.jsonl"
    output_file.write_text(
        json.dumps(
            {
                "chunk_id": "AAA_2024_Q1_001",
                "ticker": "AAA",
                "bank_name": "Alpha Bank",
                "source_type": "transcript",
                "period_year": 2024,
                "period_quarter": 1,
                "intent_level": 3,
                "intent_label": "Deploying",
                "app_categories": ["GenAI / LLMs"],
                "confidence": 0.8,
                "evidence_snippet": "existing",
            }
        )
        + "\n"
        + '{"chunk_id":"BROKEN"',
        encoding="utf-8",
    )
    progress_file = tmp_path / "classification_progress.json"

    mock_paths = MagicMock()
    mock_paths.chunks_jsonl = chunks_file
    mock_paths.classifications_jsonl = output_file
    mock_paths.classification_progress_json = progress_file

    new_record = {
        "chunk_id": "AAA_2024_Q1_002",
        "ticker": "AAA",
        "bank_name": "Alpha Bank",
        "source_type": "transcript",
        "period_year": 2024,
        "period_quarter": 1,
        "intent_level": 2,
        "intent_label": "Committing",
        "app_categories": ["Fraud / Risk Models"],
        "confidence": 0.7,
        "evidence_snippet": "new",
    }

    with patch("src.ai_corpus.intent_classifier.QwenAnswerGenerator", autospec=True), patch(
        "src.ai_corpus.intent_classifier.classify_chunk",
        return_value=new_record,
    ):
        run_classification(mock_paths, resume=True, log_every=1)

    lines = output_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert "BROKEN" not in output_file.read_text(encoding="utf-8")


def test_run_classification_skips_model_load_when_resume_is_already_complete(tmp_path: Path) -> None:
    chunks_file = tmp_path / "chunks.jsonl"
    chunk = {
        "chunk_id": "AAA_2024_Q1_001",
        "ticker": "AAA",
        "bank_name": "Alpha Bank",
        "source_type": "transcript",
        "period_year": 2024,
        "period_quarter": 1,
        "chunk_text": "We launched an AI chatbot",
        "ai_anchor": True,
    }
    chunks_file.write_text(json.dumps(chunk) + "\n", encoding="utf-8")

    output_file = tmp_path / "classifications.jsonl"
    output_file.write_text(
        json.dumps(
            {
                "chunk_id": "AAA_2024_Q1_001",
                "ticker": "AAA",
                "bank_name": "Alpha Bank",
                "source_type": "transcript",
                "period_year": 2024,
                "period_quarter": 1,
                "intent_level": 3,
                "intent_label": "Deploying",
                "app_categories": ["GenAI / LLMs"],
                "confidence": 0.8,
                "evidence_snippet": "done",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    progress_file = tmp_path / "classification_progress.json"

    mock_paths = MagicMock()
    mock_paths.chunks_jsonl = chunks_file
    mock_paths.classifications_jsonl = output_file
    mock_paths.classification_progress_json = progress_file

    with patch("src.ai_corpus.intent_classifier.QwenAnswerGenerator", autospec=True) as generator_cls:
        returned_path = run_classification(mock_paths, resume=True, log_every=1)

    generator_cls.assert_not_called()
    assert returned_path == output_file
    progress = json.loads(progress_file.read_text(encoding="utf-8"))
    assert progress["status"] == "complete"
    assert progress["processed_this_run"] == 0
    assert progress["session_throughput_chunks_per_second"] == 0.0


def test_run_classification_logs_session_scoped_resume_pace(tmp_path: Path, caplog) -> None:
    chunks_file = tmp_path / "chunks.jsonl"
    existing_record = {
        "chunk_id": "AAA_2024_Q1_001",
        "ticker": "AAA",
        "bank_name": "Alpha Bank",
        "source_type": "transcript",
        "period_year": 2024,
        "period_quarter": 1,
        "intent_level": 3,
        "intent_label": "Deploying",
        "app_categories": ["GenAI / LLMs"],
        "confidence": 0.8,
        "evidence_snippet": "existing",
    }
    pending_chunk = {
        "chunk_id": "AAA_2024_Q1_002",
        "ticker": "AAA",
        "bank_name": "Alpha Bank",
        "source_type": "transcript",
        "period_year": 2024,
        "period_quarter": 1,
        "chunk_text": "Machine learning for fraud detection",
        "ai_anchor": True,
    }
    with chunks_file.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps({**existing_record, "chunk_text": "existing", "ai_anchor": True}) + "\n")
        handle.write(json.dumps(pending_chunk) + "\n")

    output_file = tmp_path / "classifications.jsonl"
    output_file.write_text(json.dumps(existing_record) + "\n", encoding="utf-8")
    progress_file = tmp_path / "classification_progress.json"

    mock_paths = MagicMock()
    mock_paths.chunks_jsonl = chunks_file
    mock_paths.classifications_jsonl = output_file
    mock_paths.classification_progress_json = progress_file

    new_record = {
        "chunk_id": "AAA_2024_Q1_002",
        "ticker": "AAA",
        "bank_name": "Alpha Bank",
        "source_type": "transcript",
        "period_year": 2024,
        "period_quarter": 1,
        "intent_level": 2,
        "intent_label": "Committing",
        "app_categories": ["Fraud / Risk Models"],
        "confidence": 0.7,
        "evidence_snippet": "new",
    }

    caplog.set_level(logging.INFO)
    with patch("src.ai_corpus.intent_classifier.QwenAnswerGenerator", autospec=True), patch(
        "src.ai_corpus.intent_classifier.classify_chunk",
        return_value=new_record,
    ), patch(
        "src.ai_corpus.intent_classifier.time.monotonic",
        side_effect=[100.0, 104.0, 104.0],
    ):
        run_classification(mock_paths, resume=True, log_every=1)

    progress = json.loads(progress_file.read_text(encoding="utf-8"))
    assert progress["completed_chunks"] == 2
    assert progress["processed_this_run"] == 1
    assert progress["session_throughput_chunks_per_second"] == 0.25
    assert progress["throughput_chunks_per_second"] == 0.25
    assert progress["session_eta_seconds"] == 0.0
    assert progress["eta_seconds"] == 0.0
    assert "resumed=1, this-run=1" in caplog.text
    assert "session-rate=0.25 chunks/s" in caplog.text
