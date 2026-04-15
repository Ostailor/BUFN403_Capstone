from __future__ import annotations

import json
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
