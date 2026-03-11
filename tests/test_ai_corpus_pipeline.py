from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from zipfile import ZipFile

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ai_corpus.config import CorpusPaths
from src.ai_corpus.models import AskResult
from src.ai_corpus.pipeline import (
    ask,
    build_index,
    build_manifest,
    build_topic_findings,
    normalize_corpus,
    optimize_prompts,
    search,
)


def build_fixture_workspace(tmp_path: Path) -> CorpusPaths:
    roster_csv = tmp_path / "roster.csv"
    with roster_csv.open("w", encoding="utf-8", newline="") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=["Ticker", "Bank"])
        writer.writeheader()
        writer.writerow({"Ticker": "AAA", "Bank": "Alpha Bank Corp"})
        writer.writerow({"Ticker": "BBB", "Bank": "Beta Bank Inc"})

    transcript_zip = tmp_path / "transcripts.zip"
    with ZipFile(transcript_zip, "w") as archive:
        archive.writestr(
            "transcripts_final/AAA_2024_Q1.txt",
            "CEO: We are investing in artificial intelligence strategy and customer service copilots.",
        )
        archive.writestr(
            "transcripts_final/BBB_2024_Q1.txt",
            "CEO: We discussed deposits and balance sheet positioning.",
        )

    sec_zip = tmp_path / "sec.zip"
    with ZipFile(sec_zip, "w") as archive:
        archive.writestr(
            "data/sec-edgar-filings/AAA/10-K/AAA_10-K_2024_Q4/primary-document.html",
            "<html><body><h1>AI Strategy</h1><p>We are expanding generative AI in fraud and service.</p></body></html>",
        )
        archive.writestr(
            "data/sec-edgar-filings/AAA/10-Q/AAA_10-Q_2025_Q1/primary-document.html",
            "<html><body><p>Machine learning is used in compliance workflows.</p></body></html>",
        )
        archive.writestr(
            "data/sec-edgar-filings/BBB/10-K/BBB_10-K_2024_Q4/primary-document.html",
            "<html><body><p>No meaningful AI mention.</p></body></html>",
        )

    manual_dir = tmp_path / "manual_sources"
    (manual_dir / "call_reports").mkdir(parents=True)
    (manual_dir / "mra_mria").mkdir(parents=True)
    (manual_dir / "call_reports" / "AAA_call_report_2024_Q1.json").write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "CERT": 1111,
                        "REPDTE": "20240331",
                        "NAME": "ALPHA BANK",
                        "ASSET": 500000,
                        "DEP": 350000,
                        "NETINC": 12000,
                    }
                ],
                "source_url": "https://api.fdic.gov/banks/financials?filters=CERT:1111",
            }
        ),
        encoding="utf-8",
    )
    (manual_dir / "mra_mria" / "AAA_mria_2024_Q2.txt").write_text(
        "Management should strengthen AI governance and model controls.",
        encoding="utf-8",
    )

    return CorpusPaths(
        root_dir=tmp_path,
        output_dir=tmp_path / "artifacts" / "ai_corpus",
        manual_source_dir=manual_dir,
        roster_csv=roster_csv,
        sec_zip=sec_zip,
        transcript_zip=transcript_zip,
    )


def test_build_manifest_tracks_available_and_missing_sources(tmp_path: Path) -> None:
    paths = build_fixture_workspace(tmp_path)
    rows = build_manifest(paths=paths, refresh_public_catalog=False)
    frame = pd.DataFrame([row.as_dict() for row in rows])

    available_call = frame[
        (frame["ticker"] == "AAA")
        & (frame["source_type"] == "call_report")
        & (frame["period_label"] == "2024_Q1")
    ]
    assert not available_call.empty
    assert available_call.iloc[0]["observed_doc_count"] == 1

    missing_transcript = frame[
        (frame["ticker"] == "AAA")
        & (frame["source_type"] == "transcript")
        & (frame["period_label"] == "2025_Q4")
    ]
    assert not missing_transcript.empty
    assert missing_transcript.iloc[0]["status"] == "missing"

    mra_row = frame[(frame["ticker"] == "AAA") & (frame["source_type"] == "mra_mria")]
    assert not mra_row.empty
    assert mra_row.iloc[0]["status"] == "found"


def test_normalize_index_and_search_pipeline(tmp_path: Path) -> None:
    paths = build_fixture_workspace(tmp_path)
    build_manifest(paths=paths, refresh_public_catalog=False)
    normalization = normalize_corpus(paths=paths)
    assert normalization["normalized_documents"] > 0
    assert normalization["chunk_count"] > 0
    assert normalization["call_report_rows"] == 1

    index_summary = build_index(paths=paths, embedding_model="hash")
    assert index_summary["chunk_count"] > 0
    assert Path(index_summary["corpus_db"]).exists()

    result = search(
        "What does AAA say about AI strategy?",
        paths=paths,
        filters={"ticker": "AAA"},
        embedding_model="hash",
    )
    assert result["narrative_results"]
    assert all(hit["ticker"] == "AAA" for hit in result["narrative_results"])

    findings = build_topic_findings(paths=paths)
    assert findings["rows"] > 0
    assert paths.topic_findings_csv.exists()
    assert (paths.output_dir / "ai_bank_scorecard.csv").exists()
    assert (paths.output_dir / "ai_theme_examples.csv").exists()
    assert (paths.plots_dir / "ai_top_banks.png").exists()
    assert (paths.plots_dir / "ai_theme_trends_top6.png").exists()
    assert (paths.plots_dir / "ai_maturity_quadrant.png").exists()


def test_ask_uses_retrieval_and_returns_citations(tmp_path: Path, monkeypatch) -> None:
    paths = build_fixture_workspace(tmp_path)
    build_manifest(paths=paths, refresh_public_catalog=False)
    normalize_corpus(paths=paths)
    build_index(paths=paths, embedding_model="hash")

    def fake_answer(self, *, question, evidence_blocks, structured_refs, prompt_hint=""):  # type: ignore[no-untyped-def]
        return AskResult(
            answer_text="Alpha says it is investing in AI strategy and customer service.",
            citations=[chunk_id for chunk_id, _ in evidence_blocks[:2]],
            retrieved_chunk_ids=[chunk_id for chunk_id, _ in evidence_blocks],
            structured_data_refs=structured_refs,
            theme_tags=["ai_strategy", "customer_facing_ai"],
            confidence=0.88,
            metadata={"model": "test-double", "prompt_hint": prompt_hint},
        )

    monkeypatch.setattr("src.ai_corpus.pipeline.QwenAnswerGenerator.answer", fake_answer)
    result = ask(
        "What does AAA say about using AI?",
        paths=paths,
        filters={"ticker": "AAA"},
        embedding_model="hash",
    )
    assert "Alpha" in result.answer_text
    assert result.citations
    assert result.metadata["model"] == "test-double"


def test_optimize_prompts_writes_artifact(tmp_path: Path) -> None:
    paths = build_fixture_workspace(tmp_path)
    artifact = optimize_prompts(paths=paths)
    assert artifact["selected_template_name"]
    assert (paths.output_dir / "compiled_prompt.json").exists()
