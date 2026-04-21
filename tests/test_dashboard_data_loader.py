from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

pytest.importorskip("streamlit")

from dashboard.teams.ai_classification_intent import data_loader


@pytest.fixture
def fake_artifacts_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Create a tmp artifacts dir and point the loader module at it.

    Monkeypatches the `team_artifacts_dir` symbol imported into the
    loader module so callers inside `@st.cache_data`-wrapped functions
    resolve to `tmp_path` instead of the real project artifacts path.
    Caches are cleared between tests by each test as needed.
    """
    artifacts_dir = tmp_path / "artifacts" / "ai_corpus"
    artifacts_dir.mkdir(parents=True)

    def _fake_team_artifacts_dir(_reference_file: str | Path) -> Path:
        return artifacts_dir

    monkeypatch.setattr(data_loader, "team_artifacts_dir", _fake_team_artifacts_dir)
    return artifacts_dir


def _clear_all_caches() -> None:
    data_loader.load_scores.clear()
    data_loader.load_quarterly.clear()
    data_loader.load_app_categories.clear()
    data_loader.load_classifications.clear()


def test_load_scores_prefers_canonical_csv(fake_artifacts_dir: Path) -> None:
    canonical = fake_artifacts_dir / "bank_composite_scores.csv"
    canonical.write_text(
        "Ticker,Bank,Maturity,Breadth,Momentum,Composite,Rank\n"
        "AAA,Alpha Bank,75.0,60.0,10.0,65.0,1\n",
        encoding="utf-8",
    )

    _clear_all_caches()
    frame = data_loader.load_scores()

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


def test_load_classifications_reads_canonical_jsonl_and_normalizes_categories(
    fake_artifacts_dir: Path,
) -> None:
    canonical = fake_artifacts_dir / "classifications.jsonl"
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

    _clear_all_caches()
    frame = data_loader.load_classifications()

    assert list(frame["chunk_id"]) == ["AAA_001", "AAA_002"]
    assert frame.iloc[0]["app_categories"] == ["GenAI / LLMs", "Fraud / Risk Models"]
    assert frame.iloc[1]["app_categories"] == ["Predictive ML"]
    assert pd.api.types.is_numeric_dtype(frame["confidence"])


def test_dashboard_loaders_derive_from_classifications_when_csvs_are_missing(
    fake_artifacts_dir: Path,
) -> None:
    canonical = fake_artifacts_dir / "classifications.jsonl"
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

    _clear_all_caches()

    scores = data_loader.load_scores()
    quarterly = data_loader.load_quarterly()
    app_categories = data_loader.load_app_categories()

    assert list(scores["Ticker"]) == ["AAA"]
    assert {"Maturity", "Breadth", "Momentum", "Composite", "Rank"} <= set(scores.columns)
    assert list(quarterly["Ticker"]) == ["AAA", "AAA"]
    assert set(app_categories["Category"]) >= {"GenAI / LLMs", "Fraud / Risk Models"}


def test_dashboard_loaders_return_empty_frames_without_artifacts(
    fake_artifacts_dir: Path,
) -> None:
    _clear_all_caches()

    assert data_loader.load_scores().empty
    assert data_loader.load_quarterly().empty
    assert data_loader.load_app_categories().empty
    assert data_loader.load_classifications().empty


def test_dashboard_loaders_treat_empty_canonical_csv_as_empty_state(
    fake_artifacts_dir: Path,
) -> None:
    for name in [
        "bank_composite_scores.csv",
        "quarterly_progression.csv",
        "app_category_matrix.csv",
    ]:
        (fake_artifacts_dir / name).write_text("", encoding="utf-8")

    _clear_all_caches()

    assert data_loader.load_scores().empty
    assert data_loader.load_quarterly().empty
    assert data_loader.load_app_categories().empty
