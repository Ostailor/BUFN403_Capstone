from __future__ import annotations

import json
import re
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Any
from zipfile import ZipFile

import chromadb
import duckdb
import matplotlib.pyplot as plt
import pandas as pd
import requests

from .chunking import build_chunks
from .cleaning import clean_sec_html, clean_transcript_text, extract_text_from_file, split_sections
from .config import (
    DEFAULT_10K_PERIODS,
    DEFAULT_10Q_PERIODS,
    DEFAULT_AS_OF_DATE,
    DEFAULT_CALL_REPORT_PERIODS,
    DEFAULT_PROXY_PERIODS,
    DEFAULT_TRANSCRIPT_PERIODS,
    CorpusPaths,
    STRUCTURED_METRIC_ALIASES,
    THEME_KEYWORDS,
)
from .embeddings import build_embedder
from .models import (
    AcquisitionLogRow,
    AskResult,
    BankRecord,
    DiscoveredDocument,
    ManifestRow,
    NormalizedDocument,
    SearchHit,
    StructuredRef,
)
from .optimization import optimize_prompt_templates
from .public_sources import FdicClient, SecClient
from .qwen import QwenAnswerGenerator
from .themes import tag_themes
from .utils import (
    append_jsonl,
    ensure_dir,
    join_values,
    normalize_key,
    now_utc_iso,
    period_label,
    quarter_end_date,
    sha256_text,
    slugify,
    write_csv,
    write_json,
)

TRANSCRIPT_RE = re.compile(r"^transcripts_final/([A-Z0-9]+)_(\d{4})_Q([1-4])\.txt$")
SEC_HTML_RE = re.compile(
    r"^data/sec-edgar-filings/([^/]+)/([^/]+)/([^/]+)/primary-document\.html$"
)
MANUAL_PERIOD_RE = re.compile(r"([A-Z0-9]+).*?(\d{4})_Q([1-4])", re.I)
MANUAL_YEAR_ONLY_RE = re.compile(r"([A-Z0-9]+).*?(\d{4})", re.I)
PROMPT_ARTIFACT = "compiled_prompt.json"
AI_ANCHOR_PATTERNS = [
    re.compile(r"\bartificial intelligence\b", re.I),
    re.compile(r"\bmachine learning\b", re.I),
    re.compile(r"\bgenerative ai\b", re.I),
    re.compile(r"\bgenai\b", re.I),
    re.compile(r"\blarge language model(s)?\b", re.I),
    re.compile(r"\bllm(s)?\b", re.I),
    re.compile(r"\bchatbot(s)?\b", re.I),
    re.compile(r"\bcopilot(s)?\b", re.I),
    re.compile(r"\bai\b", re.I),
]


def load_bank_roster(roster_csv: Path) -> list[BankRecord]:
    frame = pd.read_csv(roster_csv)
    required = {"Ticker", "Bank"}
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(f"Roster CSV is missing required columns: {sorted(missing)}")
    frame = frame[["Ticker", "Bank"]].drop_duplicates().sort_values("Ticker")
    return [
        BankRecord(ticker=str(row.Ticker).upper(), bank_name=str(row.Bank).strip())
        for row in frame.itertuples(index=False)
    ]


def ensure_corpus_dirs(paths: CorpusPaths) -> None:
    for path in [
        paths.output_dir,
        paths.cache_dir,
        paths.normalized_dir,
        paths.documents_dir,
        paths.tables_dir,
        paths.index_dir,
        paths.plots_dir,
        paths.manual_source_dir,
        paths.manual_source_dir / "call_reports",
        paths.manual_source_dir / "mra_mria",
        paths.manual_source_dir / "transcripts",
        paths.manual_source_dir / "sec",
        paths.benchmarks_dir,
    ]:
        ensure_dir(path)


def has_ai_anchor(text: str) -> bool:
    return any(pattern.search(text) for pattern in AI_ANCHOR_PATTERNS)


def _read_zip_member(zip_path: Path, member_path: str) -> str:
    with ZipFile(zip_path) as archive:
        return archive.read(member_path).decode("utf-8", errors="ignore")


def _make_zip_ref(zip_path: Path, member_path: str) -> str:
    return f"{zip_path}::{member_path}"


def _split_zip_ref(ref: str) -> tuple[Path, str]:
    zip_name, member = ref.split("::", 1)
    return Path(zip_name), member


def discover_transcript_documents(paths: CorpusPaths, roster_map: dict[str, str]) -> list[DiscoveredDocument]:
    if not paths.transcript_zip.exists():
        return []
    docs: list[DiscoveredDocument] = []
    with ZipFile(paths.transcript_zip) as archive:
        for member in archive.namelist():
            match = TRANSCRIPT_RE.match(member)
            if not match:
                continue
            ticker, year, quarter = match.groups()
            docs.append(
                DiscoveredDocument(
                    doc_id=f"{ticker}_transcript_{year}_Q{quarter}",
                    ticker=ticker,
                    bank_name=roster_map.get(ticker, ticker),
                    source_type="transcript",
                    form_type="transcript",
                    period_year=int(year),
                    period_quarter=int(quarter),
                    filing_or_issue_date="",
                    storage_kind="zip_member",
                    source_path_or_url=_make_zip_ref(paths.transcript_zip, member),
                    local_path=_make_zip_ref(paths.transcript_zip, member),
                    content_type="text/plain",
                    metadata={"zip_member": member},
                )
            )
    return docs


def discover_sec_documents(paths: CorpusPaths, roster_map: dict[str, str]) -> list[DiscoveredDocument]:
    if not paths.sec_zip.exists():
        return []
    docs: list[DiscoveredDocument] = []
    with ZipFile(paths.sec_zip) as archive:
        for member in archive.namelist():
            match = SEC_HTML_RE.match(member)
            if not match:
                continue
            ticker, form_type, folder = match.groups()
            period_match = re.search(r"_(\d{4})_Q([1-4])", folder)
            year = int(period_match.group(1)) if period_match else None
            quarter = int(period_match.group(2)) if period_match else None
            doc_id = f"{ticker}_{slugify(form_type)}_{folder}"
            docs.append(
                DiscoveredDocument(
                    doc_id=doc_id,
                    ticker=ticker.upper(),
                    bank_name=roster_map.get(ticker.upper(), ticker.upper()),
                    source_type="sec_filing",
                    form_type=form_type,
                    period_year=year,
                    period_quarter=quarter,
                    filing_or_issue_date="",
                    storage_kind="zip_member",
                    source_path_or_url=_make_zip_ref(paths.sec_zip, member),
                    local_path=_make_zip_ref(paths.sec_zip, member),
                    content_type="text/html",
                    metadata={"zip_member": member, "folder": folder},
                )
            )
    return docs


def _infer_manual_metadata(path: Path, roster_map: dict[str, str]) -> tuple[str, int | None, int | None]:
    stem = path.stem.upper()
    match = MANUAL_PERIOD_RE.search(stem)
    if match:
        ticker, year, quarter = match.groups()
        if ticker in roster_map:
            return ticker, int(year), int(quarter)
    match = MANUAL_YEAR_ONLY_RE.search(stem)
    if match:
        ticker, year = match.groups()
        if ticker in roster_map:
            return ticker, int(year), None
    for ticker in roster_map:
        if ticker in stem:
            return ticker, None, None
    return "UNKNOWN", None, None


def discover_manual_documents(paths: CorpusPaths, roster_map: dict[str, str]) -> list[DiscoveredDocument]:
    docs: list[DiscoveredDocument] = []
    if not paths.manual_source_dir.exists():
        return docs
    for path in sorted(paths.manual_source_dir.rglob("*")):
        if not path.is_file() or path.name.startswith("."):
            continue
        if path.suffix.lower() in {".json", ".parquet"} and path.parent.name == "call_reports":
            content_type = "application/json" if path.suffix.lower() == ".json" else "application/parquet"
        else:
            content_type = "application/octet-stream"
        source_folder = path.relative_to(paths.manual_source_dir).parts[0]
        source_type = {
            "call_reports": "call_report",
            "mra_mria": "mra_mria",
            "transcripts": "transcript",
            "sec": "sec_filing",
        }.get(source_folder, source_folder.rstrip("s"))
        form_type = {
            "call_report": "call_report",
            "mra_mria": "mra_mria",
            "transcript": "transcript",
            "sec_filing": path.parent.name if path.parent.name else "manual_sec",
        }.get(source_type, source_type)
        ticker, year, quarter = _infer_manual_metadata(path, roster_map)
        docs.append(
            DiscoveredDocument(
                doc_id=f"{ticker}_{slugify(form_type)}_{sha256_text(str(path))[:12]}",
                ticker=ticker,
                bank_name=roster_map.get(ticker, ticker),
                source_type=source_type,
                form_type=form_type,
                period_year=year,
                period_quarter=quarter,
                filing_or_issue_date="",
                storage_kind="local_file",
                source_path_or_url=str(path),
                local_path=str(path),
                content_type=content_type,
                metadata={"manual_path": str(path)},
            )
        )
    return docs


def _group_docs(docs: list[DiscoveredDocument]) -> dict[tuple[str, str, str, int | None, int | None], list[DiscoveredDocument]]:
    grouped: dict[tuple[str, str, str, int | None, int | None], list[DiscoveredDocument]] = defaultdict(list)
    for doc in docs:
        grouped[(doc.ticker, doc.source_type, doc.form_type, doc.period_year, doc.period_quarter)].append(doc)
    return grouped


def _load_prompt_hint(paths: CorpusPaths) -> str:
    artifact_path = paths.output_dir / PROMPT_ARTIFACT
    if not artifact_path.exists():
        return ""
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    return str(payload.get("selected_template", "")).strip()


def _format_theme_tag(theme_tag: str) -> str:
    labels = {
        "ai_strategy": "AI strategy",
        "use_cases": "concrete use cases",
        "investment_spend": "AI investment and spend",
        "risk_controls": "risk controls",
        "governance": "governance",
        "customer_facing_ai": "customer-facing AI",
        "operations_efficiency": "operations efficiency",
        "vendors_partnerships": "vendors and partnerships",
        "measurable_outcomes": "measurable outcomes",
    }
    return labels.get(theme_tag, theme_tag.replace("_", " "))


def _join_phrases(values: list[str]) -> str:
    if not values:
        return ""
    if len(values) == 1:
        return values[0]
    if len(values) == 2:
        return f"{values[0]} and {values[1]}"
    return f"{', '.join(values[:-1])}, and {values[-1]}"


def _clean_summary_snippet(text: str, *, limit: int = 220) -> str:
    snippet = re.sub(r"\s+", " ", text).strip()
    snippet = re.sub(r"^[A-Z][A-Za-z .'\-]{1,80}:\s+", "", snippet)
    if len(snippet) > limit:
        snippet = snippet[:limit].rsplit(" ", 1)[0].rstrip(" ,;:.") + "..."
    return snippet


def _needs_summary_retry(answer: AskResult) -> bool:
    text = answer.answer_text.strip()
    if answer.confidence <= 0.2:
        return True
    if not text:
        return True
    first_line = text.splitlines()[0][:140]
    if re.match(r"^[A-Z][A-Za-z .'\-]{1,80}:\s+", first_line):
        return True
    if "?" in first_line and answer.confidence < 0.5:
        return True
    if not answer.theme_tags and len(text) < 120:
        return True
    return False


def _build_deterministic_bank_summary(
    bank: BankRecord,
    narrative_results: list[dict[str, Any]],
    structured_refs: list[dict[str, Any]],
    *,
    model_name: str,
    device: str,
) -> AskResult:
    if not narrative_results and not structured_refs:
        return AskResult(
            answer_text="insufficient evidence",
            citations=[],
            retrieved_chunk_ids=[],
            structured_data_refs=[],
            theme_tags=[],
            confidence=0.0,
            metadata={"model": model_name or "deterministic-fallback", "device": device},
        )

    theme_counts: dict[str, int] = defaultdict(int)
    snippets: list[str] = []
    citations: list[str] = []
    source_types: list[str] = []
    for row in narrative_results[:4]:
        citations.append(str(row["chunk_id"]))
        source_types.append(str(row["source_type"]))
        for theme_tag in row.get("theme_tags", []):
            theme_counts[str(theme_tag)] += 1
        snippet = _clean_summary_snippet(str(row["chunk_text"]))
        if snippet and snippet not in snippets:
            snippets.append(snippet)

    top_theme_tags = [
        theme_tag
        for theme_tag, _count in sorted(theme_counts.items(), key=lambda item: (-item[1], item[0]))[:4]
    ]
    theme_summary = _join_phrases([_format_theme_tag(theme_tag) for theme_tag in top_theme_tags])
    source_summary = _join_phrases(sorted(set(source_types)))

    parts: list[str] = []
    if theme_summary:
        parts.append(f"{bank.bank_name} discusses AI mainly in terms of {theme_summary}.")
    elif source_summary:
        parts.append(f"{bank.bank_name} has AI-related evidence in {source_summary}.")
    if snippets:
        parts.append(f"Key evidence includes: {snippets[0]}")
        if len(snippets) > 1:
            parts.append(f"Additional evidence includes: {snippets[1]}")
    elif structured_refs:
        parts.append(
            f"Structured call report data is available for {bank.ticker}, but the AI discussion is thin in narrative evidence."
        )

    return AskResult(
        answer_text=" ".join(parts).strip() or "insufficient evidence",
        citations=citations[:6],
        retrieved_chunk_ids=citations[:6],
        structured_data_refs=structured_refs[:8],
        theme_tags=top_theme_tags,
        confidence=0.35,
        metadata={"model": model_name or "deterministic-fallback", "device": device},
    )


def _prepare_period_frame(frame: pd.DataFrame) -> pd.DataFrame:
    period_frame = frame.dropna(subset=["period_year", "period_quarter"]).copy()
    if period_frame.empty:
        return period_frame
    period_frame["period_year"] = period_frame["period_year"].astype(int)
    period_frame["period_quarter"] = period_frame["period_quarter"].astype(int)
    period_frame["period_label"] = period_frame.apply(
        lambda row: period_label(int(row["period_year"]), int(row["period_quarter"])),
        axis=1,
    )
    return period_frame.sort_values(["period_year", "period_quarter"])


def _extract_focus_snippet(text: str, theme_tag: str, *, limit: int = 260) -> str:
    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return ""
    keywords = THEME_KEYWORDS.get(theme_tag, [])
    anchor_patterns = keywords + [
        "artificial intelligence",
        "generative ai",
        "machine learning",
        "genai",
        " ai ",
    ]
    lowered = f" {normalized.lower()} "
    match_pos = -1
    match_len = 0
    for keyword in anchor_patterns:
        token = keyword.lower()
        index = lowered.find(token)
        if index >= 0:
            match_pos = max(index - 1, 0)
            match_len = len(token.strip())
            break
    if match_pos < 0:
        return _clean_summary_snippet(normalized, limit=limit)
    start = max(match_pos - 80, 0)
    end = min(match_pos + max(match_len, 24) + 160, len(normalized))
    snippet = normalized[start:end].strip(" ,;:.")
    if start > 0:
        snippet = "..." + snippet
    if end < len(normalized):
        snippet = snippet.rstrip(" ,;:.") + "..."
    return snippet


def build_manifest(
    *,
    paths: CorpusPaths | None = None,
    as_of: date = DEFAULT_AS_OF_DATE,
    refresh_public_catalog: bool = True,
) -> list[ManifestRow]:
    paths = paths or CorpusPaths()
    ensure_corpus_dirs(paths)
    roster = load_bank_roster(paths.roster_csv)
    roster_map = {record.ticker: record.bank_name for record in roster}
    transcript_docs = discover_transcript_documents(paths, roster_map)
    sec_docs = discover_sec_documents(paths, roster_map)
    manual_docs = discover_manual_documents(paths, roster_map)
    all_docs = transcript_docs + sec_docs + manual_docs
    grouped = _group_docs(all_docs)

    sec_client = SecClient() if refresh_public_catalog else None
    fdic_client = FdicClient() if refresh_public_catalog else None
    rows: list[ManifestRow] = []

    for bank in roster:
        for year, quarter in DEFAULT_TRANSCRIPT_PERIODS:
            docs = grouped.get((bank.ticker, "transcript", "transcript", year, quarter), [])
            rows.append(
                ManifestRow(
                    ticker=bank.ticker,
                    bank_name=bank.bank_name,
                    source_type="transcript",
                    form_type="transcript",
                    period_year=year,
                    period_quarter=quarter,
                    period_label=period_label(year, quarter),
                    expected_doc_count=1,
                    observed_doc_count=len(docs),
                    status="available" if docs else "missing",
                    status_detail="transcript present" if docs else "earnings transcript not found locally",
                    storage_kind="zip_member" if any(doc.storage_kind == "zip_member" for doc in docs) else "local_file",
                    local_refs=join_values(doc.local_path for doc in docs),
                    source_urls="",
                    manual_search_hint=f"{bank.bank_name} earnings call transcript {year} Q{quarter}",
                    notes="",
                )
            )

        fdic_match = fdic_client.match_bank(bank.ticker, bank.bank_name) if fdic_client else None
        call_report_docs = [doc for doc in manual_docs if doc.source_type == "call_report" and doc.ticker == bank.ticker]
        call_group = _group_docs(call_report_docs)
        call_rows = fdic_client.find_call_report_rows(fdic_match.cert, DEFAULT_CALL_REPORT_PERIODS) if fdic_match else {}
        for year, quarter in DEFAULT_CALL_REPORT_PERIODS:
            docs = call_group.get((bank.ticker, "call_report", "call_report", year, quarter), [])
            row_status = "manual_review_required"
            status_detail = "FDIC institution match unavailable"
            source_url = ""
            notes = ""
            if fdic_match:
                source_url = (
                    "https://api.fdic.gov/banks/financials?"
                    f"format=json&limit=200&sort_by=REPDTE&sort_order=DESC&filters=CERT:{fdic_match.cert}"
                )
                notes = (
                    f"Matched FDIC institution '{fdic_match.institution_name}' / "
                    f"{fdic_match.holding_company} (CERT {fdic_match.cert}, confidence {fdic_match.confidence})"
                )
                if (year, quarter) in call_rows:
                    row_status = "available" if docs else "missing"
                    status_detail = "call report available via FDIC API" if docs else "call report exists via FDIC API but not stored locally"
                else:
                    row_status = "not_public"
                    status_detail = "call report not returned by FDIC API for requested quarter"
            rows.append(
                ManifestRow(
                    ticker=bank.ticker,
                    bank_name=bank.bank_name,
                    source_type="call_report",
                    form_type="call_report",
                    period_year=year,
                    period_quarter=quarter,
                    period_label=period_label(year, quarter),
                    expected_doc_count=1 if fdic_match else None,
                    observed_doc_count=len(docs),
                    status=row_status if not docs else "available",
                    status_detail=status_detail if not docs else "call report stored locally",
                    storage_kind="local_file",
                    local_refs=join_values(doc.local_path for doc in docs),
                    source_urls=source_url,
                    manual_search_hint=f"{bank.bank_name} call report {year} Q{quarter}",
                    notes=notes,
                )
            )

        sec_official_by_form: dict[str, list[Any]] = defaultdict(list)
        if sec_client:
            try:
                for filing in sec_client.list_filings(
                    bank.ticker,
                    as_of=as_of,
                    forms={"10-K", "10-Q", "8-K", "DEF 14A"},
                ):
                    sec_official_by_form[filing.form_type].append(filing)
            except Exception as exc:  # noqa: BLE001
                sec_official_by_form = defaultdict(list)
                rows.append(
                    ManifestRow(
                        ticker=bank.ticker,
                        bank_name=bank.bank_name,
                        source_type="sec_catalog",
                        form_type="catalog",
                        period_year=None,
                        period_quarter=None,
                        period_label="catalog",
                        expected_doc_count=None,
                        observed_doc_count=0,
                        status="manual_review_required",
                        status_detail=f"SEC catalog lookup failed: {exc}",
                        storage_kind="network",
                        local_refs="",
                        source_urls="https://www.sec.gov/files/company_tickers.json",
                        manual_search_hint=f"{bank.ticker} EDGAR filings",
                        notes="",
                    )
                )

        fixed_expectations = {
            "10-K": DEFAULT_10K_PERIODS,
            "10-Q": DEFAULT_10Q_PERIODS,
            "DEF 14A": DEFAULT_PROXY_PERIODS,
        }
        for form_type, periods in fixed_expectations.items():
            local_group = grouped
            filings_by_period: dict[tuple[int, int], list[Any]] = defaultdict(list)
            for filing in sec_official_by_form.get(form_type, []):
                filings_by_period[(filing.period_year, filing.period_quarter)].append(filing)
            for year, quarter in periods:
                docs = local_group.get((bank.ticker, "sec_filing", form_type, year, quarter), [])
                official_filings = filings_by_period.get((year, quarter), [])
                expected_count = len(official_filings) if sec_client and official_filings else 1
                status = "available" if len(docs) >= expected_count and expected_count > 0 else "missing"
                if sec_client and not official_filings:
                    expected_count = 0
                    status = "not_filed_by_cutoff"
                if expected_count and 0 < len(docs) < expected_count:
                    status = "partial"
                rows.append(
                    ManifestRow(
                        ticker=bank.ticker,
                        bank_name=bank.bank_name,
                        source_type="sec_filing",
                        form_type=form_type,
                        period_year=year,
                        period_quarter=quarter,
                        period_label=period_label(year, quarter),
                        expected_doc_count=expected_count,
                        observed_doc_count=len(docs),
                        status=status,
                        status_detail="local filing coverage complete" if status == "available" else "filing missing locally",
                        storage_kind="zip_member" if any(doc.storage_kind == "zip_member" for doc in docs) else "local_file",
                        local_refs=join_values(doc.local_path for doc in docs),
                        source_urls=join_values(getattr(filing, "source_url", "") for filing in official_filings),
                        manual_search_hint=f"{bank.ticker} {form_type} {year}",
                        notes="",
                    )
                )

        eight_k_filings: dict[tuple[int, int], list[Any]] = defaultdict(list)
        for filing in sec_official_by_form.get("8-K", []):
            eight_k_filings[(filing.period_year, filing.period_quarter)].append(filing)
        if not eight_k_filings:
            eight_k_local = [
                doc for doc in sec_docs
                if doc.ticker == bank.ticker and doc.form_type == "8-K"
            ]
            for doc in eight_k_local:
                eight_k_filings[(doc.period_year or 0, doc.period_quarter or 0)].append(doc)
        for (year, quarter), filings in sorted(eight_k_filings.items()):
            docs = grouped.get((bank.ticker, "sec_filing", "8-K", year, quarter), [])
            expected_count = len(filings)
            status = "available" if len(docs) >= expected_count else "partial" if docs else "missing"
            rows.append(
                ManifestRow(
                    ticker=bank.ticker,
                    bank_name=bank.bank_name,
                    source_type="sec_filing",
                    form_type="8-K",
                    period_year=year or None,
                    period_quarter=quarter or None,
                    period_label=period_label(year or None, quarter or None),
                    expected_doc_count=expected_count,
                    observed_doc_count=len(docs),
                    status=status,
                    status_detail="quarterly 8-K inventory compared to SEC filing list",
                    storage_kind="zip_member" if any(doc.storage_kind == "zip_member" for doc in docs) else "local_file",
                    local_refs=join_values(doc.local_path for doc in docs),
                    source_urls=join_values(getattr(filing, "source_url", "") for filing in filings),
                    manual_search_hint=f"{bank.ticker} 8-K {year} Q{quarter}",
                    notes="",
                )
            )

        mra_docs = [doc for doc in manual_docs if doc.source_type == "mra_mria" and doc.ticker == bank.ticker]
        rows.append(
            ManifestRow(
                ticker=bank.ticker,
                bank_name=bank.bank_name,
                source_type="mra_mria",
                form_type="mra_mria",
                period_year=None,
                period_quarter=None,
                period_label="all_periods",
                expected_doc_count=None,
                observed_doc_count=len(mra_docs),
                status="found" if mra_docs else "not_public",
                status_detail="manual or local supervisory documents found" if mra_docs else "no public MRA/MRIA document located",
                storage_kind="local_file",
                local_refs=join_values(doc.local_path for doc in mra_docs),
                source_urls="",
                manual_search_hint=f"{bank.bank_name} MRA MRIA supervisory letter AI",
                notes="MRA/MRIA inventory has no fixed expected count",
            )
        )

    write_csv([row.as_dict() for row in rows], paths.manifest_csv)
    return rows


def acquire_missing(
    *,
    paths: CorpusPaths | None = None,
    as_of: date = DEFAULT_AS_OF_DATE,
) -> list[AcquisitionLogRow]:
    paths = paths or CorpusPaths()
    ensure_corpus_dirs(paths)
    if not paths.manifest_csv.exists():
        build_manifest(paths=paths, as_of=as_of)
    frame = pd.read_csv(paths.manifest_csv)
    logs: list[AcquisitionLogRow] = []

    for row in frame.itertuples(index=False):
        if str(row.status) not in {"missing", "partial"}:
            if row.source_type == "mra_mria":
                logs.append(
                    AcquisitionLogRow(
                        timestamp_utc=now_utc_iso(),
                        ticker=row.ticker,
                        bank_name=row.bank_name,
                        source_type=row.source_type,
                        form_type=row.form_type,
                        period_label=row.period_label,
                        attempt_type="manual_hint",
                        query_or_url=row.manual_search_hint,
                        outcome="manual_review_required",
                        saved_path="",
                        notes="MRA/MRIA acquisition requires manual search or private access",
                    )
                )
            continue

        source_urls = [value.strip() for value in str(row.source_urls).split("|") if value.strip()]
        saved_path = ""
        outcome = "not_attempted"
        notes = ""
        query_or_url = source_urls[0] if source_urls else str(row.manual_search_hint)
        attempt_type = "network_download"

        if row.source_type == "call_report" and query_or_url.startswith("http"):
            response = requests.get(query_or_url, timeout=60)
            payload = response.json()
            period_repdte = quarter_end_date(int(row.period_year), int(row.period_quarter))
            matched = [
                item.get("data", {})
                for item in payload.get("data", [])
                if str(item.get("data", {}).get("REPDTE", "")) == period_repdte
            ]
            if matched:
                target = (
                    paths.manual_source_dir
                    / "call_reports"
                    / f"{row.ticker}_call_report_{row.period_label}.json"
                )
                write_json({"rows": matched, "source_url": query_or_url}, target)
                saved_path = str(target)
                outcome = "downloaded"
            else:
                outcome = "not_found"
                notes = "FDIC query returned no row for requested report date"

        elif row.source_type == "sec_filing" and source_urls:
            downloaded_paths = []
            for url in source_urls:
                try:
                    response = requests.get(
                        url,
                        timeout=60,
                        headers={"User-Agent": "BUFN403 Capstone omtailor@example.com"},
                    )
                    if response.status_code >= 400:
                        continue
                    suffix = Path(url).suffix or ".html"
                    target = (
                        paths.manual_source_dir
                        / "sec"
                        / row.form_type
                        / f"{row.ticker}_{slugify(row.form_type)}_{row.period_label}{suffix}"
                    )
                    ensure_dir(target.parent)
                    target.write_text(response.text, encoding="utf-8")
                    downloaded_paths.append(str(target))
                except Exception as exc:  # noqa: BLE001
                    notes = str(exc)
            if downloaded_paths:
                saved_path = join_values(downloaded_paths)
                outcome = "downloaded"
            else:
                outcome = "not_found"

        else:
            attempt_type = "manual_hint"
            outcome = "manual_review_required"
            notes = "No automated source URL available"

        logs.append(
            AcquisitionLogRow(
                timestamp_utc=now_utc_iso(),
                ticker=row.ticker,
                bank_name=row.bank_name,
                source_type=row.source_type,
                form_type=row.form_type,
                period_label=row.period_label,
                attempt_type=attempt_type,
                query_or_url=query_or_url,
                outcome=outcome,
                saved_path=saved_path,
                notes=notes,
            )
        )

    write_csv([log.as_dict() for log in logs], paths.acquisition_log_csv)
    build_manifest(paths=paths, as_of=as_of)
    return logs


def _read_discovered_document(doc: DiscoveredDocument) -> str:
    if doc.storage_kind == "zip_member":
        zip_path, member = _split_zip_ref(doc.local_path)
        return _read_zip_member(zip_path, member)
    return Path(doc.local_path).read_text(encoding="utf-8", errors="ignore")


def _normalize_call_report_frame(path: Path, ticker: str, bank_name: str) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload.get("rows", payload if isinstance(payload, list) else [payload])
        frame = pd.DataFrame(rows)
    elif suffix == ".csv":
        frame = pd.read_csv(path)
    elif suffix == ".parquet":
        frame = pd.read_parquet(path)
    elif suffix in {".xlsx", ".xls"}:
        frame = pd.read_excel(path)
    else:
        raise RuntimeError(f"Unsupported call report format: {path}")
    frame["ticker"] = ticker
    frame["bank_name"] = bank_name
    frame["source_path_or_url"] = str(path)
    return frame


def normalize_corpus(
    *,
    paths: CorpusPaths | None = None,
) -> dict[str, Any]:
    paths = paths or CorpusPaths()
    ensure_corpus_dirs(paths)
    roster = load_bank_roster(paths.roster_csv)
    roster_map = {record.ticker: record.bank_name for record in roster}

    transcript_docs = discover_transcript_documents(paths, roster_map)
    sec_docs = discover_sec_documents(paths, roster_map)
    manual_docs = discover_manual_documents(paths, roster_map)
    narrative_docs = [
        doc
        for doc in transcript_docs + sec_docs + manual_docs
        if doc.source_type != "call_report"
    ]
    call_report_docs = [doc for doc in manual_docs if doc.source_type == "call_report"]

    paths.chunks_jsonl.write_text("", encoding="utf-8")
    structured_frames: list[pd.DataFrame] = []
    normalized_count = 0
    chunk_count = 0

    for doc in narrative_docs:
        if doc.storage_kind == "zip_member":
            raw_text = _read_discovered_document(doc)
        else:
            raw_text = extract_text_from_file(Path(doc.local_path))
        if doc.source_type == "transcript":
            cleaned_text = clean_transcript_text(raw_text)
        elif doc.source_type == "sec_filing":
            cleaned_text = clean_sec_html(raw_text)
        else:
            cleaned_text = raw_text
        if not cleaned_text.strip():
            continue
        sections = split_sections(cleaned_text, doc.source_type)
        for section_index, (section_title, section_text) in enumerate(sections):
            normalized = NormalizedDocument(
                doc_id=f"{doc.doc_id}__sec_{section_index:03d}",
                ticker=doc.ticker,
                bank_name=doc.bank_name,
                source_type=doc.source_type,
                form_type=doc.form_type,
                period_year=doc.period_year,
                period_quarter=doc.period_quarter,
                filing_or_issue_date=doc.filing_or_issue_date,
                section_title=section_title,
                source_path_or_url=doc.source_path_or_url,
                storage_kind=doc.storage_kind,
                cleaned_text=section_text,
                theme_tags=tag_themes(section_text),
                metadata=doc.metadata | {"parent_doc_id": doc.doc_id},
            )
            write_json(normalized.as_dict(), normalized.output_path(paths.documents_dir))
            chunks = build_chunks(normalized)
            append_jsonl([chunk.as_dict() for chunk in chunks], paths.chunks_jsonl)
            normalized_count += 1
            chunk_count += len(chunks)

    for doc in call_report_docs:
        frame = _normalize_call_report_frame(Path(doc.local_path), doc.ticker, doc.bank_name)
        frame["period_year"] = doc.period_year
        frame["period_quarter"] = doc.period_quarter
        structured_frames.append(frame)

    call_report_rows = 0
    if structured_frames:
        call_report_frame = pd.concat(structured_frames, ignore_index=True)
        call_report_frame.to_parquet(paths.tables_dir / "call_reports.parquet", index=False)
        call_report_rows = len(call_report_frame)

    summary = {
        "normalized_documents": normalized_count,
        "chunk_count": chunk_count,
        "call_report_rows": call_report_rows,
        "documents_dir": str(paths.documents_dir),
        "chunks_jsonl": str(paths.chunks_jsonl),
    }
    write_json(summary, paths.output_dir / "normalization_summary.json")
    return summary


def build_index(
    *,
    paths: CorpusPaths | None = None,
    embedding_model: str | None = None,
    collection_name: str = "ai_usage_corpus",
) -> dict[str, Any]:
    paths = paths or CorpusPaths()
    ensure_corpus_dirs(paths)
    if not paths.chunks_jsonl.exists():
        normalize_corpus(paths=paths)

    chunk_rows = [
        json.loads(line)
        for line in paths.chunks_jsonl.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    chunk_rows = [row for row in chunk_rows if has_ai_anchor(row["chunk_text"])]
    embedder = build_embedder(embedding_model)
    client = chromadb.PersistentClient(path=str(paths.index_dir))
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass
    collection = client.create_collection(name=collection_name, metadata={"hnsw:space": "cosine"})
    batch_size = 64
    for start in range(0, len(chunk_rows), batch_size):
        batch = chunk_rows[start : start + batch_size]
        embeddings = embedder.encode([row["chunk_text"] for row in batch])
        metadatas = []
        for row in batch:
            metadata = {
                "doc_id": row["doc_id"],
                "ticker": row["ticker"],
                "bank_name": row["bank_name"],
                "source_type": row["source_type"],
                "form_type": row["form_type"],
                "period_year": int(row["period_year"]) if row["period_year"] is not None else -1,
                "period_quarter": int(row["period_quarter"]) if row["period_quarter"] is not None else -1,
                "filing_or_issue_date": row["filing_or_issue_date"] or "",
                "section_title": row["section_title"],
                "source_path_or_url": row["source_path_or_url"],
                "quality_flags": row["quality_flags"],
                "theme_tags": row["theme_tags"],
            }
            metadatas.append(metadata)
        collection.add(
            ids=[row["chunk_id"] for row in batch],
            embeddings=embeddings,
            documents=[row["chunk_text"] for row in batch],
            metadatas=metadatas,
        )

    con = duckdb.connect(str(paths.corpus_db))
    if (paths.tables_dir / "call_reports.parquet").exists():
        con.execute("CREATE OR REPLACE TABLE call_reports AS SELECT * FROM read_parquet(?)", [str(paths.tables_dir / "call_reports.parquet")])
    if paths.manifest_csv.exists():
        con.execute("CREATE OR REPLACE TABLE manifest AS SELECT * FROM read_csv_auto(?)", [str(paths.manifest_csv)])
    con.close()
    summary = {
        "collection_name": collection_name,
        "chunk_count": len(chunk_rows),
        "embedding_model": getattr(embedder, "model_name", "hash"),
        "index_dir": str(paths.index_dir),
        "corpus_db": str(paths.corpus_db),
    }
    write_json(summary, paths.output_dir / "index_summary.json")
    return summary


def _extract_ticker(question: str, filters: dict[str, Any] | None, roster: list[BankRecord]) -> str | None:
    if filters and filters.get("ticker"):
        return str(filters["ticker"]).upper()
    tickers = {record.ticker for record in roster}
    for token in re.findall(r"\b[A-Z]{2,5}\b", question.upper()):
        if token in tickers:
            return token
    lowered = question.lower()
    for record in roster:
        if record.bank_name.lower() in lowered:
            return record.ticker
    return None


def _structured_refs_from_call_reports(
    *,
    question: str,
    paths: CorpusPaths,
    roster: list[BankRecord],
    filters: dict[str, Any] | None,
) -> list[StructuredRef]:
    if not paths.corpus_db.exists():
        return []
    ticker = _extract_ticker(question, filters, roster)
    metric_names = [
        metric
        for metric, aliases in STRUCTURED_METRIC_ALIASES.items()
        if any(alias in question.lower() for alias in aliases)
    ]
    if not ticker and not metric_names and "call report" not in question.lower():
        return []
    con = duckdb.connect(str(paths.corpus_db), read_only=True)
    try:
        frame = con.execute("SELECT * FROM call_reports").df()
    except Exception:
        con.close()
        return []
    con.close()
    if ticker:
        frame = frame[frame["ticker"] == ticker]
    if frame.empty:
        return []
    frame = frame.sort_values("REPDTE", ascending=False).head(4)
    refs: list[StructuredRef] = []
    metric_column_map = {
        "assets": "ASSET",
        "deposits": "DEP",
        "net_income": "NETINC",
        "roa": "ROA",
        "roe": "ROE",
        "equity": "EQ",
        "nim": "NIMY",
    }
    if not metric_names:
        metric_names = ["assets", "deposits", "net_income"]
    for metric_name in metric_names:
        column = metric_column_map.get(metric_name)
        if not column or column not in frame.columns:
            continue
        for row in frame.itertuples(index=False):
            refs.append(
                StructuredRef(
                    ticker=str(row.ticker),
                    report_date=str(getattr(row, "REPDTE", "")),
                    cert=int(getattr(row, "CERT", 0)) if getattr(row, "CERT", None) else None,
                    metric_name=metric_name,
                    metric_value=getattr(row, column),
                    source_url=str(getattr(row, "source_path_or_url", "")),
                )
            )
    return refs


def search(
    question: str,
    *,
    paths: CorpusPaths | None = None,
    filters: dict[str, Any] | None = None,
    embedding_model: str | None = None,
    embedder_instance: Any | None = None,
    collection_name: str = "ai_usage_corpus",
    n_results: int = 8,
) -> dict[str, Any]:
    paths = paths or CorpusPaths()
    ensure_corpus_dirs(paths)
    if not (paths.output_dir / "index_summary.json").exists():
        build_index(paths=paths, embedding_model=embedding_model, collection_name=collection_name)
    roster = load_bank_roster(paths.roster_csv)
    embedder = embedder_instance or build_embedder(embedding_model)
    query_embedding = embedder.encode([question])[0]
    client = chromadb.PersistentClient(path=str(paths.index_dir))
    collection = client.get_collection(collection_name)
    where = {}
    if filters:
        if filters.get("ticker"):
            where["ticker"] = str(filters["ticker"]).upper()
        if filters.get("source_type"):
            where["source_type"] = str(filters["source_type"])
        if filters.get("form_type"):
            where["form_type"] = str(filters["form_type"])
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        where=where or None,
    )
    hits: list[SearchHit] = []
    ids = results.get("ids", [[]])[0]
    docs = results.get("documents", [[]])[0]
    distances = results.get("distances", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    for index, chunk_id in enumerate(ids):
        metadata = metadatas[index]
        hits.append(
            SearchHit(
                rank=index + 1,
                chunk_id=chunk_id,
                score=1.0 - float(distances[index]),
                ticker=str(metadata.get("ticker", "")),
                bank_name=str(metadata.get("bank_name", "")),
                source_type=str(metadata.get("source_type", "")),
                form_type=str(metadata.get("form_type", "")),
                period_year=int(metadata.get("period_year")) if int(metadata.get("period_year", -1)) >= 0 else None,
                period_quarter=int(metadata.get("period_quarter")) if int(metadata.get("period_quarter", -1)) >= 0 else None,
                section_title=str(metadata.get("section_title", "")),
                source_path_or_url=str(metadata.get("source_path_or_url", "")),
                theme_tags=[value for value in str(metadata.get("theme_tags", "")).split(",") if value],
                chunk_text=docs[index],
            )
        )
    structured_refs = _structured_refs_from_call_reports(
        question=question,
        paths=paths,
        roster=roster,
        filters=filters,
    )
    return {
        "question": question,
        "narrative_results": [hit.as_dict() for hit in hits],
        "structured_results": [ref.as_dict() for ref in structured_refs],
    }


def ask(
    question: str,
    *,
    paths: CorpusPaths | None = None,
    filters: dict[str, Any] | None = None,
    embedding_model: str | None = None,
    collection_name: str = "ai_usage_corpus",
) -> AskResult:
    paths = paths or CorpusPaths()
    prompt_hint = _load_prompt_hint(paths)
    search_result = search(
        question,
        paths=paths,
        filters=filters,
        embedding_model=embedding_model,
        collection_name=collection_name,
    )
    evidence_blocks = [
        (row["chunk_id"], row["chunk_text"])
        for row in search_result["narrative_results"][:5]
    ]
    structured_refs = search_result["structured_results"][:8]
    generator = QwenAnswerGenerator()
    return generator.answer(
        question=question,
        evidence_blocks=evidence_blocks,
        structured_refs=structured_refs,
        prompt_hint=prompt_hint,
    )


def optimize_prompts(
    *,
    paths: CorpusPaths | None = None,
    benchmark_path: Path | None = None,
) -> dict[str, Any]:
    paths = paths or CorpusPaths()
    ensure_corpus_dirs(paths)
    benchmark_path = benchmark_path or (paths.benchmarks_dir / "bank_ai_questions.jsonl")
    if not benchmark_path.exists():
        benchmark_examples = [
            {
                "question": "What does BAC say about AI strategy and customer-facing use cases?",
                "expected_terms": ["AI", "strategy", "customer"],
                "expected_theme_tags": ["ai_strategy", "customer_facing_ai"],
                "should_abstain": False,
            },
            {
                "question": "What does WFC say about AI governance and controls?",
                "expected_terms": ["governance", "controls", "AI"],
                "expected_theme_tags": ["risk_controls", "governance"],
                "should_abstain": False,
            },
            {
                "question": "Does the corpus support a claim that every bank has a public MRIA about AI?",
                "expected_terms": ["insufficient evidence"],
                "expected_theme_tags": [],
                "should_abstain": True,
            },
        ]
        ensure_dir(benchmark_path.parent)
        benchmark_path.write_text(
            "\n".join(json.dumps(row) for row in benchmark_examples),
            encoding="utf-8",
        )
    artifact = optimize_prompt_templates(
        benchmark_path,
        output_path=paths.output_dir / PROMPT_ARTIFACT,
    )
    return artifact


def build_topic_findings(
    *,
    paths: CorpusPaths | None = None,
) -> dict[str, Any]:
    paths = paths or CorpusPaths()
    ensure_corpus_dirs(paths)
    if not paths.chunks_jsonl.exists():
        normalize_corpus(paths=paths)
    frame = pd.read_json(paths.chunks_jsonl, lines=True)
    if frame.empty:
        summary = {"topic_findings_csv": str(paths.topic_findings_csv), "rows": 0}
        write_json(summary, paths.output_dir / "topic_findings_summary.json")
        return summary
    frame = frame[frame["chunk_text"].map(has_ai_anchor)].copy()
    if frame.empty:
        summary = {"topic_findings_csv": str(paths.topic_findings_csv), "rows": 0}
        write_json(summary, paths.output_dir / "topic_findings_summary.json")
        return summary
    exploded = frame.assign(theme_tag=frame["theme_tags"].fillna("").str.split(",")).explode("theme_tag")
    exploded = exploded[exploded["theme_tag"].fillna("") != ""].copy()
    findings = (
        exploded.groupby(
            ["ticker", "bank_name", "source_type", "period_year", "period_quarter", "theme_tag"],
            dropna=False,
        )
        .size()
        .reset_index(name="mention_count")
        .sort_values(["ticker", "period_year", "period_quarter", "mention_count"], ascending=[True, True, True, False])
    )
    findings.to_csv(paths.topic_findings_csv, index=False)

    ensure_dir(paths.plots_dir)
    generated_plots: list[str] = []
    generated_tables: list[str] = [str(paths.topic_findings_csv)]

    theme_totals = findings.groupby("theme_tag")["mention_count"].sum().sort_values(ascending=False)
    theme_totals_csv = paths.output_dir / "ai_theme_totals.csv"
    theme_totals.reset_index().to_csv(theme_totals_csv, index=False)
    generated_tables.append(str(theme_totals_csv))

    if not theme_totals.empty:
        ordered_theme_totals = theme_totals.sort_values(ascending=True)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(
            [_format_theme_tag(value) for value in ordered_theme_totals.index],
            ordered_theme_totals.values,
            color="#0b7285",
        )
        ax.set_title("AI Theme Mentions Across Corpus")
        ax.set_xlabel("Mention count")
        ax.grid(axis="x", alpha=0.2)
        fig.tight_layout()
        theme_totals_plot = paths.plots_dir / "ai_theme_totals.png"
        fig.savefig(theme_totals_plot, dpi=170)
        plt.close(fig)
        generated_plots.append(str(theme_totals_plot))

    bank_totals = (
        findings.groupby(["ticker", "bank_name"])["mention_count"]
        .sum()
        .reset_index(name="total_mentions")
        .sort_values("total_mentions", ascending=False)
    )
    bank_rankings_csv = paths.output_dir / "ai_bank_rankings.csv"
    bank_totals.to_csv(bank_rankings_csv, index=False)
    generated_tables.append(str(bank_rankings_csv))

    if not bank_totals.empty:
        top_banks = bank_totals.head(20).sort_values("total_mentions", ascending=True)
        fig, ax = plt.subplots(figsize=(11, 8))
        ax.barh(top_banks["ticker"], top_banks["total_mentions"], color="#2b8a3e")
        ax.set_title("Top Banks by AI Mention Count")
        ax.set_xlabel("Theme mentions")
        ax.grid(axis="x", alpha=0.2)
        fig.tight_layout()
        top_banks_plot = paths.plots_dir / "ai_top_banks.png"
        fig.savefig(top_banks_plot, dpi=170)
        plt.close(fig)
        generated_plots.append(str(top_banks_plot))

    mentions_by_quarter = (
        findings.groupby(["period_year", "period_quarter"], dropna=False)["mention_count"]
        .sum()
        .reset_index()
        .sort_values(["period_year", "period_quarter"])
    )
    if not mentions_by_quarter.empty:
        labels = [period_label(int(row.period_year), int(row.period_quarter)) for row in mentions_by_quarter.itertuples()]
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(labels, mentions_by_quarter["mention_count"], marker="o", color="#0b7285")
        ax.set_title("AI Theme Mentions by Quarter")
        ax.set_ylabel("Mention count")
        ax.tick_params(axis="x", rotation=45)
        ax.grid(alpha=0.2)
        fig.tight_layout()
        mentions_plot = paths.plots_dir / "ai_mentions_by_quarter.png"
        fig.savefig(mentions_plot, dpi=170)
        plt.close(fig)
        generated_plots.append(str(mentions_plot))

    heatmap = (
        findings.groupby(["ticker", "theme_tag"])["mention_count"]
        .sum()
        .reset_index()
        .pivot(index="ticker", columns="theme_tag", values="mention_count")
        .fillna(0)
        .reindex(columns=list(theme_totals.index), fill_value=0)
    )
    if not heatmap.empty:
        fig, ax = plt.subplots(figsize=(12, 10))
        image = ax.imshow(heatmap.values, aspect="auto", cmap="Blues")
        ax.set_xticks(range(len(heatmap.columns)))
        ax.set_xticklabels([_format_theme_tag(value) for value in heatmap.columns], rotation=45, ha="right")
        ax.set_yticks(range(len(heatmap.index)))
        ax.set_yticklabels(heatmap.index)
        ax.set_title("Bank AI Theme Heatmap")
        fig.colorbar(image, ax=ax, shrink=0.8)
        fig.tight_layout()
        bank_theme_heatmap = paths.plots_dir / "bank_theme_heatmap.png"
        fig.savefig(bank_theme_heatmap, dpi=170)
        plt.close(fig)
        generated_plots.append(str(bank_theme_heatmap))

    period_findings = _prepare_period_frame(findings)
    quarterly_theme = (
        period_findings.groupby(["period_year", "period_quarter", "period_label", "theme_tag"])["mention_count"]
        .sum()
        .reset_index()
        .sort_values(["period_year", "period_quarter", "theme_tag"])
    )
    quarterly_theme_csv = paths.output_dir / "ai_theme_quarterly_trends.csv"
    quarterly_theme.to_csv(quarterly_theme_csv, index=False)
    generated_tables.append(str(quarterly_theme_csv))

    if not quarterly_theme.empty:
        top_theme_names = list(theme_totals.head(6).index)
        fig, ax = plt.subplots(figsize=(11, 6))
        for theme_tag in top_theme_names:
            theme_frame = quarterly_theme[quarterly_theme["theme_tag"] == theme_tag]
            if theme_frame.empty:
                continue
            ax.plot(
                theme_frame["period_label"],
                theme_frame["mention_count"],
                marker="o",
                linewidth=2,
                label=_format_theme_tag(theme_tag),
            )
        ax.set_title("Top AI Themes by Quarter")
        ax.set_ylabel("Mention count")
        ax.tick_params(axis="x", rotation=45)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8, ncol=2)
        fig.tight_layout()
        theme_trends_plot = paths.plots_dir / "ai_theme_trends_top6.png"
        fig.savefig(theme_trends_plot, dpi=170)
        plt.close(fig)
        generated_plots.append(str(theme_trends_plot))

        quarter_heatmap = quarterly_theme.pivot_table(
            index="period_label",
            columns="theme_tag",
            values="mention_count",
            aggfunc="sum",
            fill_value=0,
        ).reindex(columns=list(theme_totals.index), fill_value=0)
        if not quarter_heatmap.empty:
            fig, ax = plt.subplots(figsize=(12, 7))
            image = ax.imshow(quarter_heatmap.values, aspect="auto", cmap="YlGnBu")
            ax.set_xticks(range(len(quarter_heatmap.columns)))
            ax.set_xticklabels([_format_theme_tag(value) for value in quarter_heatmap.columns], rotation=45, ha="right")
            ax.set_yticks(range(len(quarter_heatmap.index)))
            ax.set_yticklabels(quarter_heatmap.index)
            ax.set_title("AI Theme Intensity by Quarter")
            fig.colorbar(image, ax=ax, shrink=0.8)
            fig.tight_layout()
            period_heatmap_plot = paths.plots_dir / "ai_theme_heatmap_by_quarter.png"
            fig.savefig(period_heatmap_plot, dpi=170)
            plt.close(fig)
            generated_plots.append(str(period_heatmap_plot))

    source_mix = findings.groupby("source_type")["mention_count"].sum().sort_values(ascending=False)
    source_mix_csv = paths.output_dir / "ai_source_mix.csv"
    source_mix.reset_index().to_csv(source_mix_csv, index=False)
    generated_tables.append(str(source_mix_csv))
    if not source_mix.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(source_mix.index, source_mix.values, color="#2b8a3e")
        ax.set_title("AI Evidence by Source Type")
        ax.set_ylabel("Mention count")
        ax.tick_params(axis="x", rotation=30)
        ax.grid(axis="y", alpha=0.2)
        fig.tight_layout()
        source_plot = paths.plots_dir / "ai_evidence_by_source.png"
        fig.savefig(source_plot, dpi=170)
        plt.close(fig)
        generated_plots.append(str(source_plot))

    source_theme = (
        findings.groupby(["source_type", "theme_tag"])["mention_count"]
        .sum()
        .reset_index()
        .pivot(index="source_type", columns="theme_tag", values="mention_count")
        .fillna(0)
        .reindex(columns=list(theme_totals.index), fill_value=0)
    )
    source_theme_csv = paths.output_dir / "ai_source_theme_matrix.csv"
    source_theme.reset_index().to_csv(source_theme_csv, index=False)
    generated_tables.append(str(source_theme_csv))
    if not source_theme.empty:
        fig, ax = plt.subplots(figsize=(11, 4.5))
        image = ax.imshow(source_theme.values, aspect="auto", cmap="Greens")
        ax.set_xticks(range(len(source_theme.columns)))
        ax.set_xticklabels([_format_theme_tag(value) for value in source_theme.columns], rotation=45, ha="right")
        ax.set_yticks(range(len(source_theme.index)))
        ax.set_yticklabels(source_theme.index)
        ax.set_title("AI Themes by Source Type")
        fig.colorbar(image, ax=ax, shrink=0.8)
        fig.tight_layout()
        source_theme_plot = paths.plots_dir / "ai_source_theme_heatmap.png"
        fig.savefig(source_theme_plot, dpi=170)
        plt.close(fig)
        generated_plots.append(str(source_theme_plot))

    if not period_findings.empty:
        source_quarter = (
            period_findings.groupby(["period_year", "period_quarter", "period_label", "source_type"])["mention_count"]
            .sum()
            .reset_index()
        )
        source_quarter_pivot = source_quarter.pivot_table(
            index="period_label",
            columns="source_type",
            values="mention_count",
            aggfunc="sum",
            fill_value=0,
        )
        source_quarter_csv = paths.output_dir / "ai_source_quarterly_trends.csv"
        source_quarter_pivot.reset_index().to_csv(source_quarter_csv, index=False)
        generated_tables.append(str(source_quarter_csv))
        if not source_quarter_pivot.empty:
            fig, ax = plt.subplots(figsize=(10, 6))
            bottom = pd.Series([0] * len(source_quarter_pivot), index=source_quarter_pivot.index, dtype="float64")
            palette = ["#1c7ed6", "#2b8a3e", "#e8590c", "#862e9c"]
            for idx, column in enumerate(source_quarter_pivot.columns):
                values = source_quarter_pivot[column]
                ax.bar(
                    source_quarter_pivot.index,
                    values,
                    bottom=bottom,
                    label=column,
                    color=palette[idx % len(palette)],
                )
                bottom = bottom + values
            ax.set_title("AI Evidence by Quarter and Source")
            ax.set_ylabel("Mention count")
            ax.tick_params(axis="x", rotation=45)
            ax.legend()
            ax.grid(axis="y", alpha=0.2)
            fig.tight_layout()
            source_quarter_plot = paths.plots_dir / "ai_sources_by_quarter.png"
            fig.savefig(source_quarter_plot, dpi=170)
            plt.close(fig)
            generated_plots.append(str(source_quarter_plot))

    bank_theme_counts = (
        findings.groupby(["ticker", "bank_name", "theme_tag"])["mention_count"]
        .sum()
        .reset_index()
        .pivot(index=["ticker", "bank_name"], columns="theme_tag", values="mention_count")
        .fillna(0)
        .reset_index()
    )
    for theme_tag in theme_totals.index:
        if theme_tag not in bank_theme_counts.columns:
            bank_theme_counts[theme_tag] = 0

    source_counts = (
        findings.groupby(["ticker", "bank_name", "source_type"])["mention_count"]
        .sum()
        .reset_index()
        .pivot(index=["ticker", "bank_name"], columns="source_type", values="mention_count")
        .fillna(0)
        .reset_index()
    )
    bank_scorecard = bank_theme_counts.merge(source_counts, on=["ticker", "bank_name"], how="left").fillna(0)
    theme_columns = [value for value in theme_totals.index if value in bank_scorecard.columns]
    bank_scorecard["total_mentions"] = bank_scorecard[theme_columns].sum(axis=1)
    bank_scorecard["governance_strategy_mentions"] = bank_scorecard[
        [value for value in ["ai_strategy", "governance", "risk_controls"] if value in bank_scorecard.columns]
    ].sum(axis=1)
    bank_scorecard["execution_mentions"] = bank_scorecard[
        [
            value
            for value in [
                "use_cases",
                "operations_efficiency",
                "customer_facing_ai",
                "investment_spend",
                "vendors_partnerships",
                "measurable_outcomes",
            ]
            if value in bank_scorecard.columns
        ]
    ].sum(axis=1)
    denominator = bank_scorecard["total_mentions"].replace(0, pd.NA)
    bank_scorecard["governance_strategy_share"] = (
        bank_scorecard["governance_strategy_mentions"] / denominator
    ).fillna(0.0)
    bank_scorecard["execution_share"] = (bank_scorecard["execution_mentions"] / denominator).fillna(0.0)

    latest_assets = pd.DataFrame()
    call_report_path = paths.tables_dir / "call_reports.parquet"
    if call_report_path.exists():
        call_reports = pd.read_parquet(call_report_path)
        if {"ticker", "ASSET"}.issubset(call_reports.columns):
            latest_assets = (
                call_reports.sort_values("REPDTE")
                .groupby("ticker")
                .tail(1)[["ticker", "ASSET"]]
            )
            mentions = findings.groupby("ticker")["mention_count"].sum().reset_index()
            merged = latest_assets.merge(mentions, on="ticker", how="inner")
            if not merged.empty:
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.scatter(merged["ASSET"], merged["mention_count"], color="#c92a2a", alpha=0.8)
                for row in merged.itertuples(index=False):
                    ax.annotate(row.ticker, (row.ASSET, row.mention_count), fontsize=8)
                ax.set_title("Latest Assets vs AI Mention Count")
                ax.set_xlabel("Assets")
                ax.set_ylabel("AI mention count")
                ax.grid(alpha=0.2)
                fig.tight_layout()
                assets_plot = paths.plots_dir / "assets_vs_ai_mentions.png"
                fig.savefig(assets_plot, dpi=170)
                plt.close(fig)
                generated_plots.append(str(assets_plot))

                merged["mentions_per_billion_matched_assets"] = merged["mention_count"] / (
                    merged["ASSET"] / 1_000_000_000
                )
                intensity = merged.sort_values("mentions_per_billion_matched_assets", ascending=False).head(15)
                if not intensity.empty:
                    intensity = intensity.sort_values("mentions_per_billion_matched_assets", ascending=True)
                    fig, ax = plt.subplots(figsize=(11, 7))
                    ax.barh(intensity["ticker"], intensity["mentions_per_billion_matched_assets"], color="#c92a2a")
                    ax.set_title("AI Mention Intensity per $1B Matched Call-Report Assets")
                    ax.set_xlabel("Mentions per $1B matched call-report assets")
                    ax.grid(axis="x", alpha=0.2)
                    fig.tight_layout()
                    intensity_plot = paths.plots_dir / "ai_mentions_per_billion_matched_assets.png"
                    fig.savefig(intensity_plot, dpi=170)
                    plt.close(fig)
                    generated_plots.append(str(intensity_plot))
        if not latest_assets.empty:
            bank_scorecard = bank_scorecard.merge(latest_assets, on="ticker", how="left")
            bank_scorecard["mentions_per_billion_matched_assets"] = (
                bank_scorecard["total_mentions"] / (bank_scorecard["ASSET"] / 1_000_000_000)
            ).replace([float("inf")], pd.NA)

    bank_scorecard = bank_scorecard.sort_values("total_mentions", ascending=False)
    bank_scorecard_csv = paths.output_dir / "ai_bank_scorecard.csv"
    bank_scorecard.to_csv(bank_scorecard_csv, index=False)
    generated_tables.append(str(bank_scorecard_csv))

    if not bank_scorecard.empty:
        top_scorecard = bank_scorecard.head(12).copy().sort_values("total_mentions", ascending=True)
        fig, ax = plt.subplots(figsize=(12, 8))
        bottom = pd.Series([0] * len(top_scorecard), index=top_scorecard.index, dtype="float64")
        theme_palette = [
            "#0b7285",
            "#1c7ed6",
            "#2b8a3e",
            "#74b816",
            "#e67700",
            "#c92a2a",
            "#ae3ec9",
            "#495057",
            "#f08c00",
        ]
        for idx, theme_tag in enumerate(theme_columns):
            values = top_scorecard[theme_tag]
            ax.barh(
                top_scorecard["ticker"],
                values,
                left=bottom,
                label=_format_theme_tag(theme_tag),
                color=theme_palette[idx % len(theme_palette)],
            )
            bottom = bottom + values
        ax.set_title("AI Theme Mix for Top Banks")
        ax.set_xlabel("Theme mentions")
        ax.legend(fontsize=8, ncol=2)
        ax.grid(axis="x", alpha=0.2)
        fig.tight_layout()
        theme_mix_plot = paths.plots_dir / "ai_top_banks_theme_mix.png"
        fig.savefig(theme_mix_plot, dpi=170)
        plt.close(fig)
        generated_plots.append(str(theme_mix_plot))

        fig, ax = plt.subplots(figsize=(10, 7))
        ax.scatter(
            top_scorecard["governance_strategy_share"],
            top_scorecard["execution_share"],
            s=top_scorecard["total_mentions"].clip(lower=1) * 3,
            alpha=0.75,
            color="#5f3dc4",
            edgecolors="white",
            linewidths=0.8,
        )
        for row in top_scorecard.itertuples(index=False):
            ax.annotate(row.ticker, (row.governance_strategy_share, row.execution_share), fontsize=8)
        ax.set_title("AI Governance vs Execution Focus")
        ax.set_xlabel("Governance + strategy share")
        ax.set_ylabel("Execution + outcome share")
        ax.grid(alpha=0.2)
        fig.tight_layout()
        quadrant_plot = paths.plots_dir / "ai_maturity_quadrant.png"
        fig.savefig(quadrant_plot, dpi=170)
        plt.close(fig)
        generated_plots.append(str(quadrant_plot))

    example_rows: list[dict[str, Any]] = []
    example_candidates = _prepare_period_frame(exploded)
    if not example_candidates.empty:
        example_candidates["snippet"] = example_candidates.apply(
            lambda row: _extract_focus_snippet(str(row["chunk_text"]), str(row["theme_tag"]), limit=260),
            axis=1,
        )
        example_candidates["source_priority"] = example_candidates["source_type"].map(
            {"transcript": 0, "sec_filing": 1, "mra_mria": 2, "call_report": 3}
        ).fillna(9)
        example_candidates["text_length"] = example_candidates["chunk_text"].str.len()
        example_candidates = example_candidates.sort_values(
            ["theme_tag", "source_priority", "period_year", "period_quarter", "text_length"],
            ascending=[True, True, False, False, False],
        )
        for theme_tag, theme_frame in example_candidates.groupby("theme_tag"):
            distinct = theme_frame.drop_duplicates(subset=["ticker"]).head(5)
            for row in distinct.itertuples(index=False):
                example_rows.append(
                    {
                        "theme_tag": theme_tag,
                        "theme_label": _format_theme_tag(str(theme_tag)),
                        "ticker": row.ticker,
                        "bank_name": row.bank_name,
                        "source_type": row.source_type,
                        "period_year": int(row.period_year),
                        "period_quarter": int(row.period_quarter),
                        "chunk_id": row.chunk_id,
                        "section_title": row.section_title,
                        "snippet": row.snippet,
                        "source_path_or_url": row.source_path_or_url,
                    }
                )
    examples_csv = paths.output_dir / "ai_theme_examples.csv"
    pd.DataFrame(example_rows).to_csv(examples_csv, index=False)
    generated_tables.append(str(examples_csv))

    report_path = paths.output_dir / "ai_visual_report.md"
    report_lines = [
        "# AI Visual Analysis Pack",
        "",
        f"- Banks with AI evidence: {int(bank_totals['ticker'].nunique())}",
        f"- Total AI theme mentions: {int(theme_totals.sum())}",
        f"- Dominant sources: {', '.join(f'{source} ({count})' for source, count in source_mix.head(3).items())}",
        f"- Top themes: {', '.join(_format_theme_tag(value) for value in theme_totals.head(5).index)}",
        f"- Top banks by AI mention count: {', '.join(bank_totals.head(10)['ticker'].tolist())}",
        "",
        "## Generated plots",
    ]
    report_lines.extend(f"- {path}" for path in generated_plots)
    report_lines.extend(["", "## Generated tables"])
    report_lines.extend(f"- {path}" for path in generated_tables)
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    generated_tables.append(str(report_path))

    summary = {
        "topic_findings_csv": str(paths.topic_findings_csv),
        "rows": int(len(findings)),
        "plots_dir": str(paths.plots_dir),
        "generated_plots": generated_plots,
        "generated_tables": generated_tables,
        "report_path": str(report_path),
    }
    write_json(summary, paths.output_dir / "topic_findings_summary.json")
    return summary


def build_bank_ai_summaries(
    *,
    paths: CorpusPaths | None = None,
    embedding_model: str | None = None,
    collection_name: str = "ai_usage_corpus",
) -> dict[str, Any]:
    paths = paths or CorpusPaths()
    ensure_corpus_dirs(paths)
    if not (paths.output_dir / "index_summary.json").exists():
        build_index(paths=paths, embedding_model=embedding_model, collection_name=collection_name)
    if not (paths.output_dir / PROMPT_ARTIFACT).exists():
        optimize_prompts(paths=paths)

    roster = load_bank_roster(paths.roster_csv)
    prompt_hint = _load_prompt_hint(paths)
    generator = QwenAnswerGenerator()
    shared_embedder = build_embedder(embedding_model)
    con = duckdb.connect(str(paths.corpus_db), read_only=True) if paths.corpus_db.exists() else None

    summary_rows: list[dict[str, Any]] = []
    for bank in roster:
        query = (
            f"What does {bank.bank_name} ({bank.ticker}) say about using AI? "
            "Summarize AI strategy, use cases, investment/spend, governance or controls, "
            "customer-facing AI, operations efficiency, vendor/partnership mentions, and measurable outcomes. "
            "If the evidence is weak, say insufficient evidence."
        )
        search_result = search(
            query,
            paths=paths,
            filters={"ticker": bank.ticker},
            embedding_model=embedding_model,
            embedder_instance=shared_embedder,
            collection_name=collection_name,
        )
        evidence_blocks = [
            (row["chunk_id"], row["chunk_text"])
            for row in search_result["narrative_results"][:6]
        ]
        structured_refs = search_result["structured_results"][:8]
        answer = generator.answer(
            question=query,
            evidence_blocks=evidence_blocks,
            structured_refs=structured_refs,
            prompt_hint=prompt_hint,
        )
        if _needs_summary_retry(answer):
            retry_query = (
                f"What concrete AI use cases, investments, controls, partnerships, or outcomes does "
                f"{bank.bank_name} ({bank.ticker}) describe? Use only the evidence and provide a concise synthesis."
            )
            retry_search = search(
                retry_query,
                paths=paths,
                filters={"ticker": bank.ticker},
                embedding_model=embedding_model,
                embedder_instance=shared_embedder,
                collection_name=collection_name,
            )
            retry_evidence_blocks = [
                (row["chunk_id"], row["chunk_text"])
                for row in retry_search["narrative_results"][:4]
            ]
            retry_structured_refs = retry_search["structured_results"][:8]
            retry_answer = generator.answer(
                question=retry_query,
                evidence_blocks=retry_evidence_blocks,
                structured_refs=retry_structured_refs,
                prompt_hint=prompt_hint,
            )
            if _needs_summary_retry(retry_answer):
                effective_results = retry_search if retry_search["narrative_results"] else search_result
                effective_structured = retry_structured_refs or structured_refs
                answer = _build_deterministic_bank_summary(
                    bank,
                    effective_results["narrative_results"],
                    effective_structured,
                    model_name=retry_answer.metadata.get("model", "") or answer.metadata.get("model", ""),
                    device=retry_answer.metadata.get("device", "") or answer.metadata.get("device", ""),
                )
            else:
                search_result = retry_search
                answer = retry_answer

        latest_call = None
        if con is not None:
            try:
                latest_call = con.execute(
                    """
                    SELECT REPDTE, ASSET, DEP, NETINC
                    FROM call_reports
                    WHERE ticker = ?
                    ORDER BY REPDTE DESC
                    LIMIT 1
                    """,
                    [bank.ticker],
                ).fetchone()
            except Exception:
                latest_call = None

        summary_rows.append(
            {
                "Ticker": bank.ticker,
                "Bank": bank.bank_name,
                "AI_Summary": answer.answer_text,
                "Confidence": round(answer.confidence, 2),
                "Theme_Tags": ",".join(answer.theme_tags),
                "Citations": " | ".join(answer.citations),
                "Retrieved_Chunk_Count": len(answer.retrieved_chunk_ids),
                "Structured_Ref_Count": len(answer.structured_data_refs),
                "Latest_Call_Report_Date": latest_call[0] if latest_call else "",
                "Latest_Assets": latest_call[1] if latest_call else "",
                "Latest_Deposits": latest_call[2] if latest_call else "",
                "Latest_Net_Income": latest_call[3] if latest_call else "",
                "Model": answer.metadata.get("model", ""),
                "Device": answer.metadata.get("device", ""),
            }
        )

    if con is not None:
        con.close()

    summary_frame = pd.DataFrame(summary_rows).sort_values("Ticker")
    csv_path = paths.output_dir / "bank_ai_summaries.csv"
    json_path = paths.output_dir / "bank_ai_summaries.json"
    md_path = paths.output_dir / "bank_ai_summaries.md"
    summary_frame.to_csv(csv_path, index=False)
    summary_frame.to_json(json_path, orient="records", indent=2)

    lines = ["# Bank AI Summaries", ""]
    for row in summary_frame.itertuples(index=False):
        lines.append(f"## {row.Ticker} - {row.Bank}")
        lines.append(f"- Summary: {row.AI_Summary}")
        lines.append(f"- Confidence: {row.Confidence}")
        if row.Theme_Tags:
            lines.append(f"- Themes: {row.Theme_Tags}")
        if row.Citations:
            lines.append(f"- Citations: {row.Citations}")
        if row.Latest_Call_Report_Date:
            lines.append(
                f"- Latest call report: {row.Latest_Call_Report_Date}; "
                f"assets={row.Latest_Assets}, deposits={row.Latest_Deposits}, net_income={row.Latest_Net_Income}"
            )
        lines.append("")
    md_path.write_text("\n".join(lines), encoding="utf-8")

    summary = {
        "bank_count": int(len(summary_frame)),
        "csv_path": str(csv_path),
        "json_path": str(json_path),
        "markdown_path": str(md_path),
        "model": summary_frame["Model"].mode().iloc[0] if not summary_frame.empty else "",
    }
    write_json(summary, paths.output_dir / "bank_ai_summaries_summary.json")
    return summary
