from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "classify_bank_ai.py"
SPEC = importlib.util.spec_from_file_location("classify_bank_ai", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_normalize_company_name() -> None:
    assert MODULE.normalize_company_name("BANK OF AMERICA CORP /DE/") == "Bank Of America Corp"
    assert MODULE.normalize_company_name("CITIGROUP INC") == "Citigroup Inc"


def test_clean_10k_html_removes_hidden_and_tags() -> None:
    raw = """
    <html>
      <body>
        <div style=\"display:none\">us-gaap:valuationtechniqueoptionpricingmodelmember</div>
        <p>We invest in artificial intelligence for fraud detection.</p>
      </body>
    </html>
    """
    cleaned = MODULE.clean_10k_html(raw)
    assert "artificial intelligence" in cleaned.lower()
    assert "valuationtechniqueoptionpricingmodelmember" not in cleaned.lower()
    assert "<p>" not in cleaned


def test_analyze_rule_signals_positive_doc() -> None:
    text = "We are investing in artificial intelligence strategy and machine learning fraud detection."
    config = MODULE.load_config(ROOT / "config" / "scoring.yml")
    result = MODULE.analyze_rule_signals(text, config)
    assert result.signal_count >= 1
    assert result.rule_score >= 2.0
    assert result.anchor_hits >= 1


def test_apply_percentile_normalization_bounds() -> None:
    rows = [
        MODULE.BankOutputRow(
            Bank="A",
            Ticker="A",
            AI_Score=1.0,
            Rule_AI_Score=1.0,
            LLM_AI_Score=1.0,
            AI_Score_Normalized=0.0,
            Evidence="x",
            Evidence_Source="none",
            signal_count=0,
            llm_confidence=0.0,
            missing_sources="none",
            num_transcripts=1,
            num_10k_docs=1,
        ),
        MODULE.BankOutputRow(
            Bank="B",
            Ticker="B",
            AI_Score=10.0,
            Rule_AI_Score=10.0,
            LLM_AI_Score=10.0,
            AI_Score_Normalized=0.0,
            Evidence="x",
            Evidence_Source="none",
            signal_count=0,
            llm_confidence=0.0,
            missing_sources="none",
            num_transcripts=1,
            num_10k_docs=1,
        ),
    ]
    MODULE.apply_percentile_normalization(rows)
    assert rows[0].AI_Score_Normalized == 1.0
    assert rows[1].AI_Score_Normalized == 10.0


def test_calibrate_llm_scores_expands_compressed_distribution() -> None:
    config = MODULE.load_config(ROOT / "config" / "scoring.yml")
    rows = [
        MODULE.BankOutputRow(
            Bank=f"B{i}",
            Ticker=f"T{i}",
            AI_Score=4.0,
            Rule_AI_Score=rule,
            LLM_AI_Score=2.0 + (i % 2) * 0.1,
            AI_Score_Normalized=1.0,
            Evidence="x",
            Evidence_Source="none",
            signal_count=1,
            llm_confidence=0.6,
            missing_sources="none",
            num_transcripts=2,
            num_10k_docs=2,
        )
        for i, rule in enumerate([2.0, 2.5, 3.0, 4.0, 5.5, 6.0, 7.0, 8.0], start=1)
    ]

    before_span = max(r.LLM_AI_Score for r in rows) - min(r.LLM_AI_Score for r in rows)
    MODULE.calibrate_llm_scores(rows, config, verbose=False)
    after_span = max(r.LLM_AI_Score for r in rows) - min(r.LLM_AI_Score for r in rows)

    assert before_span <= 0.2
    assert after_span > before_span
    assert all(1.0 <= r.AI_Score <= 10.0 for r in rows)


def test_choose_backend_hf_local_without_token() -> None:
    args = argparse.Namespace(
        provider="huggingface",
        hf_model="Qwen/Qwen2.5-0.5B-Instruct",
        hf_timeout=30,
        hf_retries=1,
        hf_local=True,
        hf_local_device="cpu",
        hf_local_max_new_tokens=32,
        hf_local_max_input_tokens=256,
        hf_token_env="HF_TOKEN",
        allow_rule_fallback=False,
        gemini_model="gemini-2.5-pro",
    )
    backend = MODULE.choose_backend(args)
    assert backend is not None
    assert isinstance(backend, MODULE.HuggingFaceBackend)
    assert backend.local is True
    assert backend.token is None


def test_arg_parser_accepts_hf_local_flags() -> None:
    parser = MODULE.build_arg_parser()
    args = parser.parse_args(
        [
            "--sec-zip",
            "/tmp/sec.zip",
            "--transcript-zip",
            "/tmp/transcript.zip",
            "--provider",
            "huggingface",
            "--hf-local",
            "--hf-local-device",
            "cpu",
            "--hf-local-max-new-tokens",
            "64",
            "--hf-local-max-input-tokens",
            "512",
        ]
    )
    assert args.hf_local is True
    assert args.hf_local_device == "cpu"
    assert args.hf_local_max_new_tokens == 64
    assert args.hf_local_max_input_tokens == 512


def test_integration_fixture_pipeline(tmp_path: Path) -> None:
    transcript_zip = tmp_path / "transcripts.zip"
    sec_zip = tmp_path / "sec.zip"

    with ZipFile(transcript_zip, "w") as zf:
        zf.writestr(
            "transcripts_final/AAA_2025_Q4.txt",
            "We deployed artificial intelligence for fraud detection and customer service.",
        )
        zf.writestr(
            "transcripts_final/BBB_2025_Q4.txt",
            "This quarter focused on deposits and branch operations without AI detail.",
        )
        zf.writestr(
            "transcripts_final/CCC_2025_Q4.txt",
            "Machine learning initiatives support risk modeling and underwriting.",
        )

    with ZipFile(sec_zip, "w") as zf:
        for ticker, name, html_content in [
            (
                "AAA",
                "ALPHA BANK CORP /DE/",
                "<html><body><p>Generative AI investment program is expanding.</p></body></html>",
            ),
            (
                "BBB",
                "BETA BANK INC",
                "<html><body><p>The filing discusses liquidity and rates.</p></body></html>",
            ),
            (
                "CCC",
                "CHARLIE FINANCIAL CORP",
                "<html><body><p>AI and machine learning models are used in compliance.</p></body></html>",
            ),
        ]:
            zf.writestr(
                f"data/sec-edgar-filings/{ticker}/10-K/{ticker}_10-K_2025_Q4/full-submission.txt",
                f"COMPANY CONFORMED NAME:\t\t\t{name}\n",
            )
            zf.writestr(
                f"data/sec-edgar-filings/{ticker}/10-K/{ticker}_10-K_2025_Q4/primary-document.html",
                html_content,
            )

    out_csv = tmp_path / "AI_Bank_Classification.csv"
    out_xlsx = tmp_path / "AI_Bank_Classification.xlsx"

    rows = MODULE.run_pipeline(
        sec_zip_path=sec_zip,
        transcript_zip_path=transcript_zip,
        config_path=ROOT / "config" / "scoring.yml",
        backend=None,
        out_csv=out_csv,
        out_xlsx=out_xlsx,
        verbose=False,
        max_banks=None,
    )

    assert len(rows) == 3
    assert out_csv.exists()
    assert out_xlsx.exists()
    assert all(row.Evidence for row in rows)
    scores = {row.Ticker: row.AI_Score for row in rows}
    assert scores["AAA"] >= scores["BBB"]
