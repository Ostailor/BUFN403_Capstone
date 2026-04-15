from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dashboard import data_loader


def _make_reference_file(tmp_path: Path) -> tuple[Path, Path]:
    project_root = tmp_path / "project"
    pages_dir = project_root / "dashboard" / "pages"
    artifacts_dir = project_root / "artifacts" / "ai_corpus"
    pages_dir.mkdir(parents=True)
    artifacts_dir.mkdir(parents=True)
    reference_file = pages_dir / "test_page.py"
    reference_file.write_text("# test page\n", encoding="utf-8")
    return reference_file, artifacts_dir


def test_load_scores_prefers_canonical_csv(tmp_path: Path) -> None:
    reference_file, artifacts_dir = _make_reference_file(tmp_path)
    canonical = artifacts_dir / "bank_composite_scores.csv"
    canonical.write_text(
        "Ticker,Bank,Maturity,Breadth,Momentum,Composite,Rank\n"
        "AAA,Alpha Bank,75.0,60.0,10.0,65.0,1\n",
        encoding="utf-8",
    )

    data_loader.load_scores.clear()
    frame = data_loader.load_scores(str(reference_file))

    assert frame.to_dict("records") == [
        {
            "Ticker": "AAA",
            "Bank": "Alpha Bank",
            "Maturity": 75.0,
            "Breadth": 60.0,
            "Momentum": 10.0,
            "Composite": 65.0,
            "Rank": 1,
        }
    ]


def test_load_classifications_reads_canonical_jsonl_and_normalizes_categories(tmp_path: Path) -> None:
    reference_file, artifacts_dir = _make_reference_file(tmp_path)
    canonical = artifacts_dir / "classifications.jsonl"
    records = [
        {
            "chunk_id": "AAA_001",
            "ticker": "AAA",
            "bank_name": "Alpha Bank",
            "source_type": "transcript",
            "period_year": 2024,
            "period_quarter": 1,
            "intent_level": 3,
            "intent_label": "Deploying",
            "app_categories": ["GenAI / LLMs", "Fraud / Risk Models"],
            "confidence": 0.91,
            "evidence_snippet": "Alpha deployed a GenAI fraud workflow.",
        },
        {
            "chunk_id": "AAA_002",
            "ticker": "AAA",
            "bank_name": "Alpha Bank",
            "source_type": "10-K",
            "period_year": 2024,
            "period_quarter": 2,
            "intent_level": 2,
            "intent_label": "Committing",
            "app_categories": json.dumps(["Predictive ML"]),
            "confidence": 0.83,
            "evidence_snippet": "Alpha is investing in predictive models.",
        },
    ]
    canonical.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )

    data_loader.load_classifications.clear()
    frame = data_loader.load_classifications(str(reference_file))

    assert list(frame["chunk_id"]) == ["AAA_001", "AAA_002"]
    assert frame.iloc[0]["app_categories"] == ["GenAI / LLMs", "Fraud / Risk Models"]
    assert frame.iloc[1]["app_categories"] == ["Predictive ML"]
    assert pd.api.types.is_numeric_dtype(frame["confidence"])


def test_dashboard_loaders_derive_from_classifications_when_csvs_are_missing(tmp_path: Path) -> None:
    reference_file, artifacts_dir = _make_reference_file(tmp_path)
    canonical = artifacts_dir / "classifications.jsonl"
    canonical.write_text(
        "\n".join(
            [
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
                        "confidence": 0.91,
                        "evidence_snippet": "Alpha deployed copilots.",
                    }
                ),
                json.dumps(
                    {
                        "chunk_id": "AAA_002",
                        "ticker": "AAA",
                        "bank_name": "Alpha Bank",
                        "source_type": "10-K",
                        "period_year": 2024,
                        "period_quarter": 2,
                        "intent_level": 4,
                        "intent_label": "Scaling",
                        "app_categories": ["Fraud / Risk Models"],
                        "confidence": 0.88,
                        "evidence_snippet": "Alpha scaled fraud AI.",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    data_loader.load_scores.clear()
    data_loader.load_quarterly.clear()
    data_loader.load_app_categories.clear()

    scores = data_loader.load_scores(str(reference_file))
    quarterly = data_loader.load_quarterly(str(reference_file))
    app_categories = data_loader.load_app_categories(str(reference_file))

    assert list(scores["Ticker"]) == ["AAA"]
    assert {"Maturity", "Breadth", "Momentum", "Composite", "Rank"} <= set(scores.columns)
    assert list(quarterly["Ticker"]) == ["AAA", "AAA"]
    assert set(app_categories["Category"]) >= {"GenAI / LLMs", "Fraud / Risk Models"}


def test_dashboard_loaders_return_empty_frames_without_artifacts(tmp_path: Path) -> None:
    reference_file, _ = _make_reference_file(tmp_path)

    data_loader.load_scores.clear()
    data_loader.load_quarterly.clear()
    data_loader.load_app_categories.clear()
    data_loader.load_classifications.clear()

    assert data_loader.load_scores(str(reference_file)).empty
    assert data_loader.load_quarterly(str(reference_file)).empty
    assert data_loader.load_app_categories(str(reference_file)).empty
    assert data_loader.load_classifications(str(reference_file)).empty


def test_dashboard_loaders_treat_empty_canonical_csv_as_empty_state(tmp_path: Path) -> None:
    reference_file, artifacts_dir = _make_reference_file(tmp_path)
    for name in [
        "bank_composite_scores.csv",
        "quarterly_progression.csv",
        "app_category_matrix.csv",
    ]:
        (artifacts_dir / name).write_text("", encoding="utf-8")

    data_loader.load_scores.clear()
    data_loader.load_quarterly.clear()
    data_loader.load_app_categories.clear()

    assert data_loader.load_scores(str(reference_file)).empty
    assert data_loader.load_quarterly(str(reference_file)).empty
    assert data_loader.load_app_categories(str(reference_file)).empty
