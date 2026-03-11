#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import html
import os
import re
import statistics
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable
from zipfile import ZipFile

import yaml
from openpyxl import Workbook

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.llm.backends.base import BackendError, LLMBackend
from src.ai_corpus.cleaning import clean_sec_html as shared_clean_sec_html
from src.ai_corpus.cleaning import clean_transcript_text as shared_clean_transcript_text
from src.llm.backends.gemini_backend import GeminiBackend
from src.llm.backends.huggingface_backend import HuggingFaceBackend

TRANSCRIPT_RE = re.compile(r"^transcripts_final/([A-Z0-9]+)_(\d{4})_Q([1-4])\.txt$")
SEC_10K_HTML_RE = re.compile(
    r"^data/sec-edgar-filings/([^/]+)/10-K/[^/]*_10-K_(\d{4})_Q([1-4])/primary-document\.html$"
)
SEC_10K_FULL_RE = re.compile(
    r"^data/sec-edgar-filings/([^/]+)/10-K/[^/]*_10-K_(\d{4})_Q([1-4])/full-submission\.txt$"
)

ANCHOR_PATTERNS = [
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

EXPLICIT_TERMS = {
    "strategy",
    "strategic",
    "invest",
    "investment",
    "deploy",
    "deployed",
    "deployment",
    "initiative",
    "program",
    "platform",
    "capability",
    "budget",
    "spend",
    "roadmap",
}

USE_CASE_TERMS = {
    "fraud",
    "risk model",
    "risk modeling",
    "underwriting",
    "trading",
    "customer service",
    "chatbot",
    "copilot",
    "personalization",
    "compliance",
    "anti-money laundering",
    "aml",
    "credit decision",
    "contact center",
    "operations",
}

WEAK_TERMS = {
    "automation",
    "productivity",
    "efficiency",
    "assistant",
    "tooling",
}

NON_AI_PATTERNS = [
    re.compile(r"option pricing model", re.I),
    re.compile(r"valuation technique option pricing model", re.I),
    re.compile(r"valuationtechniqueoptionpricingmodelmember", re.I),
    re.compile(r"us-gaap:[^\s]{0,120}modelmember", re.I),
]

DEFAULT_CONFIG: dict[str, Any] = {
    "prefilter": {
        "anchors": [
            "artificial intelligence",
            "machine learning",
            "generative ai",
            "genai",
            "large language model",
            "llm",
            "chatbot",
            "copilot",
            "ai",
        ],
        "llm_min_anchor_hits": 1,
    },
    "weights": {
        "explicit": 3.0,
        "use_case": 2.0,
        "weak": 1.0,
        "non_ai_penalty": -1.5,
    },
    "scoring": {
        "min_score": 1.0,
        "max_score": 10.0,
        "doc_rule_base": 1.0,
        "doc_rule_multiplier": 0.6,
        "doc_weight_rule": 0.6,
        "doc_weight_llm": 0.4,
        "bank_weight_transcript": 0.65,
        "bank_weight_10k": 0.35,
        "missing_source_penalty": 0.3,
        "llm_calibration_enabled": True,
        "llm_compression_span_threshold": 2.0,
        "llm_compression_std_threshold": 0.55,
        "llm_target_std_factor": 0.8,
        "llm_target_std_floor": 0.9,
        "llm_calibration_rule_blend": 0.3,
        "max_hybrid_gap_below_rule_when_llm_compressed": 0.5,
    },
    "text_limits": {
        "max_chars_transcript": 12000,
        "max_chars_10k": 10000,
        "max_evidence_chars": 220,
    },
}


@dataclass(slots=True)
class DocumentRef:
    ticker: str
    bank_name: str
    doc_type: str
    period: str
    zip_member_path: str


@dataclass(slots=True)
class DocumentScore:
    ticker: str
    doc_id: str
    doc_type: str
    period: str
    rule_score: float
    llm_score: float
    final_score: float
    evidence_summary: str
    confidence: float
    signal_count: int


@dataclass(slots=True)
class BankOutputRow:
    Bank: str
    Ticker: str
    AI_Score: float
    Rule_AI_Score: float
    LLM_AI_Score: float
    AI_Score_Normalized: float
    Evidence: str
    Evidence_Source: str
    signal_count: int
    llm_confidence: float
    missing_sources: str
    num_transcripts: int
    num_10k_docs: int


@dataclass(slots=True)
class RuleAnalysis:
    rule_points: float
    rule_score: float
    signal_count: int
    anchor_hits: int
    evidence_summary: str


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def safe_mean(values: Iterable[float], default: float) -> float:
    values_list = list(values)
    if not values_list:
        return default
    return statistics.fmean(values_list)


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(config_path: Path | None) -> dict[str, Any]:
    config = dict(DEFAULT_CONFIG)
    if not config_path or not config_path.exists():
        return config
    with config_path.open("r", encoding="utf-8") as infile:
        loaded = yaml.safe_load(infile) or {}
    if not isinstance(loaded, dict):
        return config
    return deep_merge(config, loaded)


def period_key(period: str) -> tuple[int, int]:
    year, quarter = period.split("_Q")
    return int(year), int(quarter)


def normalize_company_name(raw_name: str) -> str:
    name = html.unescape(raw_name).strip()
    name = re.sub(r"\s*/[A-Z]{1,4}/\s*$", "", name)
    name = re.sub(r"\s+", " ", name)
    if name.isupper():
        lowered = name.lower().title()
        replacements = {
            "Llc": "LLC",
            "Inc": "Inc",
            "Corp": "Corp",
            "Co": "Co",
            "N.A": "N.A",
            "Na": "N.A",
            "&": "&",
        }
        parts = []
        for token in lowered.split(" "):
            parts.append(replacements.get(token, token))
        name = " ".join(parts)
    return name


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    cleaned = [p.strip() for p in parts if p and p.strip()]
    return cleaned


def clean_transcript_text(text: str) -> str:
    return shared_clean_transcript_text(text)


def clean_10k_html(text: str) -> str:
    return shared_clean_sec_html(text)


def count_anchor_hits(text: str) -> int:
    hits = 0
    for pattern in ANCHOR_PATTERNS:
        hits += len(pattern.findall(text))
    return hits


def summarize_sentence(sentence: str, category: str, max_chars: int) -> str:
    sentence_lower = sentence.lower()
    if not sentence_lower.strip():
        return "No clear AI initiative is detailed in this document."[:max_chars]

    ai_markers = [
        "generative ai",
        "machine learning",
        "artificial intelligence",
        "llm",
        "chatbot",
        "copilot",
        "ai",
    ]
    action_markers = [
        "investment",
        "invest",
        "deployment",
        "deploy",
        "initiative",
        "platform",
        "strategy",
        "program",
    ]
    use_case_markers = [
        "fraud",
        "risk",
        "trading",
        "underwriting",
        "customer service",
        "personalization",
        "compliance",
        "operations",
    ]

    found_ai = [marker for marker in ai_markers if marker in sentence_lower]
    found_actions = [marker for marker in action_markers if marker in sentence_lower]
    found_use_cases = [marker for marker in use_case_markers if marker in sentence_lower]

    ai_label = found_ai[0] if found_ai else "AI"
    action_label = found_actions[0] if found_actions else "initiative"

    if category == "explicit":
        if found_use_cases:
            summary = f"Highlights {ai_label} {action_label} tied to {', '.join(found_use_cases[:2])}."
        else:
            summary = f"Highlights explicit {ai_label} {action_label} in the business strategy."
    elif category == "use_case":
        if found_use_cases:
            summary = f"Mentions {ai_label} deployment for {', '.join(found_use_cases[:3])}."
        else:
            summary = "Mentions concrete AI/ML deployment in business use cases."
    elif category == "weak":
        summary = f"References early {ai_label} automation/productivity efforts with limited detail."
    else:
        summary = "No clear AI initiative is detailed in this document."

    return summary[:max_chars]


def analyze_rule_signals(text: str, config: dict[str, Any]) -> RuleAnalysis:
    text_lower = text.lower()
    scoring_cfg = config["scoring"]
    weights = config["weights"]

    sentences = split_sentences(text)
    rule_points = 0.0
    signal_count = 0
    best_sentence = ""
    best_category = "none"
    category_priority = {"none": 0, "weak": 1, "use_case": 2, "explicit": 3}

    for sentence in sentences:
        sentence_lower = sentence.lower()
        has_anchor = any(pattern.search(sentence_lower) for pattern in ANCHOR_PATTERNS)
        if not has_anchor:
            continue

        explicit_hit = any(term in sentence_lower for term in EXPLICIT_TERMS)
        use_case_hit = any(term in sentence_lower for term in USE_CASE_TERMS)
        weak_hit = any(term in sentence_lower for term in WEAK_TERMS)

        category = "none"
        if explicit_hit:
            rule_points += float(weights["explicit"])
            signal_count += 1
            category = "explicit"
        elif use_case_hit:
            rule_points += float(weights["use_case"])
            signal_count += 1
            category = "use_case"
        elif weak_hit:
            rule_points += float(weights["weak"])
            signal_count += 1
            category = "weak"

        if category_priority[category] > category_priority[best_category]:
            best_category = category
            best_sentence = sentence

    penalty_hits = 0
    for pattern in NON_AI_PATTERNS:
        if pattern.search(text_lower):
            penalty_hits += 1
    if penalty_hits:
        rule_points += float(weights["non_ai_penalty"]) * min(penalty_hits, 3)

    rule_points = clamp(rule_points, -6.0, 15.0)
    rule_score = clamp(
        float(scoring_cfg["doc_rule_base"]) + float(scoring_cfg["doc_rule_multiplier"]) * rule_points,
        float(scoring_cfg["min_score"]),
        float(scoring_cfg["max_score"]),
    )

    evidence_summary = summarize_sentence(
        best_sentence,
        best_category,
        int(config["text_limits"]["max_evidence_chars"]),
    )
    anchor_hits = count_anchor_hits(text_lower)

    return RuleAnalysis(
        rule_points=rule_points,
        rule_score=rule_score,
        signal_count=signal_count,
        anchor_hits=anchor_hits,
        evidence_summary=evidence_summary,
    )


def build_llm_context(text: str, max_chars: int) -> str:
    sentences = split_sentences(text)
    if not sentences:
        return text[:max_chars]

    selected_indices: set[int] = set()
    for index, sentence in enumerate(sentences):
        sentence_lower = sentence.lower()
        if any(pattern.search(sentence_lower) for pattern in ANCHOR_PATTERNS):
            selected_indices.add(index)
            if index - 1 >= 0:
                selected_indices.add(index - 1)
            if index + 1 < len(sentences):
                selected_indices.add(index + 1)

    if not selected_indices:
        context = " ".join(sentences[:20])
        return context[:max_chars]

    ordered = [sentences[i] for i in sorted(selected_indices)]
    context = " ".join(ordered)
    if len(context) < 500:
        context = " ".join(sentences[: min(len(sentences), 40)])
    return context[:max_chars]


def parse_company_name_from_full_submission(text: str) -> str | None:
    match = re.search(r"^\s*COMPANY CONFORMED NAME:\s*(.+?)\s*$", text, flags=re.M)
    if not match:
        return None
    return normalize_company_name(match.group(1))


def choose_backend(args: argparse.Namespace) -> LLMBackend | None:
    if args.provider == "none":
        return None

    if args.provider == "huggingface":
        model = args.hf_model
        if args.hf_local and model == "meta-llama/Llama-3.1-8B-Instruct":
            model = "google/flan-t5-small"
            print(
                "Using local mode with a lighter default model: "
                f"{model} (override with --hf-model).",
                file=sys.stderr,
            )

        if args.hf_local:
            return HuggingFaceBackend(
                token=None,
                model=model,
                timeout_seconds=args.hf_timeout,
                max_retries=args.hf_retries,
                local=True,
                local_device=args.hf_local_device,
                local_max_new_tokens=args.hf_local_max_new_tokens,
                local_max_input_tokens=args.hf_local_max_input_tokens,
            )

        token = os.getenv(args.hf_token_env, "")
        if not token:
            if args.allow_rule_fallback:
                print(
                    "HF token missing; running with deterministic rule-only fallback.",
                    file=sys.stderr,
                )
                return None
            raise SystemExit(
                f"Environment variable '{args.hf_token_env}' is required for provider 'huggingface'."
            )
        return HuggingFaceBackend(
            token=token,
            model=model,
            timeout_seconds=args.hf_timeout,
            max_retries=args.hf_retries,
            local=False,
        )

    if args.provider == "gemini":
        gemini = GeminiBackend(api_key=os.getenv("GEMINI_API_KEY"), model=args.gemini_model)
        if args.allow_rule_fallback:
            print(
                "Gemini backend is scaffold-only; running with deterministic rule-only fallback.",
                file=sys.stderr,
            )
            return None
        return gemini

    raise SystemExit(f"Unsupported provider: {args.provider}")


def index_transcript_docs(transcript_zip: ZipFile) -> dict[str, list[DocumentRef]]:
    by_ticker: dict[str, list[DocumentRef]] = defaultdict(list)
    for member in transcript_zip.namelist():
        match = TRANSCRIPT_RE.match(member)
        if not match:
            continue
        ticker, year, quarter = match.groups()
        by_ticker[ticker].append(
            DocumentRef(
                ticker=ticker,
                bank_name="",
                doc_type="transcript",
                period=f"{year}_Q{quarter}",
                zip_member_path=member,
            )
        )
    for ticker in by_ticker:
        by_ticker[ticker].sort(key=lambda d: period_key(d.period))
    return by_ticker


def index_sec_docs(sec_zip: ZipFile) -> tuple[dict[str, list[DocumentRef]], dict[str, dict[str, str]]]:
    html_by_ticker: dict[str, list[DocumentRef]] = defaultdict(list)
    full_by_ticker: dict[str, dict[str, str]] = defaultdict(dict)

    for member in sec_zip.namelist():
        html_match = SEC_10K_HTML_RE.match(member)
        if html_match:
            ticker, year, quarter = html_match.groups()
            period = f"{year}_Q{quarter}"
            html_by_ticker[ticker].append(
                DocumentRef(
                    ticker=ticker,
                    bank_name="",
                    doc_type="10-K",
                    period=period,
                    zip_member_path=member,
                )
            )
            continue

        full_match = SEC_10K_FULL_RE.match(member)
        if full_match:
            ticker, year, quarter = full_match.groups()
            period = f"{year}_Q{quarter}"
            full_by_ticker[ticker][period] = member

    for ticker in html_by_ticker:
        html_by_ticker[ticker].sort(key=lambda d: period_key(d.period))

    return html_by_ticker, full_by_ticker


def derive_bank_names(sec_zip: ZipFile, full_by_ticker: dict[str, dict[str, str]]) -> dict[str, str]:
    names: dict[str, str] = {}
    for ticker, period_map in full_by_ticker.items():
        if not period_map:
            continue
        latest_period = max(period_map.keys(), key=period_key)
        member = period_map[latest_period]
        raw_text = sec_zip.read(member).decode("utf-8", errors="ignore")
        parsed = parse_company_name_from_full_submission(raw_text)
        if parsed:
            names[ticker] = parsed
        else:
            names[ticker] = ticker
    return names


def score_document(
    doc: DocumentRef,
    cleaned_text: str,
    backend: LLMBackend | None,
    config: dict[str, Any],
    verbose: bool,
) -> DocumentScore:
    scoring_cfg = config["scoring"]
    text_limit_key = "max_chars_10k" if doc.doc_type == "10-K" else "max_chars_transcript"
    max_chars = int(config["text_limits"][text_limit_key])

    rule = analyze_rule_signals(cleaned_text, config)
    llm_score = rule.rule_score
    confidence = clamp(0.35 + 0.07 * min(rule.signal_count, 5) + 0.03 * min(rule.anchor_hits, 5), 0.0, 1.0)
    evidence_summary = rule.evidence_summary

    min_anchor_hits = int(config["prefilter"]["llm_min_anchor_hits"])
    should_call_llm = backend is not None and rule.anchor_hits >= min_anchor_hits
    if should_call_llm:
        llm_context = build_llm_context(cleaned_text, max_chars=max_chars)
        try:
            llm_result = backend.score_document(
                llm_context,
                ticker=doc.ticker,
                doc_type=doc.doc_type,
                period=doc.period,
            )
            llm_score = clamp(
                llm_result.score,
                float(scoring_cfg["min_score"]),
                float(scoring_cfg["max_score"]),
            )
            confidence = clamp(llm_result.confidence, 0.0, 1.0)
            if llm_result.evidence_summary:
                evidence_summary = llm_result.evidence_summary[: int(config["text_limits"]["max_evidence_chars"])]
        except BackendError as exc:
            if verbose:
                print(
                    f"LLM scoring failed for {doc.ticker} {doc.doc_type} {doc.period}: {exc}",
                    file=sys.stderr,
                )

    final_score = round(
        clamp(
            float(scoring_cfg["doc_weight_rule"]) * rule.rule_score
            + float(scoring_cfg["doc_weight_llm"]) * llm_score,
            float(scoring_cfg["min_score"]),
            float(scoring_cfg["max_score"]),
        ),
        1,
    )

    return DocumentScore(
        ticker=doc.ticker,
        doc_id=Path(doc.zip_member_path).name,
        doc_type=doc.doc_type,
        period=doc.period,
        rule_score=round(rule.rule_score, 1),
        llm_score=round(llm_score, 1),
        final_score=final_score,
        evidence_summary=evidence_summary,
        confidence=round(confidence, 2),
        signal_count=rule.signal_count,
    )


def aggregate_bank(
    ticker: str,
    bank_name: str,
    transcript_scores: list[DocumentScore],
    tenk_scores: list[DocumentScore],
    config: dict[str, Any],
) -> BankOutputRow:
    scoring_cfg = config["scoring"]
    min_score = float(scoring_cfg["min_score"])
    max_score = float(scoring_cfg["max_score"])

    missing: list[str] = []
    if not transcript_scores:
        missing.append("transcript")
    if not tenk_scores:
        missing.append("10-K")

    transcript_mean = safe_mean([d.final_score for d in transcript_scores], min_score)
    tenk_mean = safe_mean([d.final_score for d in tenk_scores], min_score)
    rule_transcript_mean = safe_mean([d.rule_score for d in transcript_scores], min_score)
    rule_tenk_mean = safe_mean([d.rule_score for d in tenk_scores], min_score)
    llm_transcript_mean = safe_mean([d.llm_score for d in transcript_scores], min_score)
    llm_tenk_mean = safe_mean([d.llm_score for d in tenk_scores], min_score)
    missing_count = len(missing)

    ai_score = round(
        clamp(
            float(scoring_cfg["bank_weight_transcript"]) * transcript_mean
            + float(scoring_cfg["bank_weight_10k"]) * tenk_mean
            - float(scoring_cfg["missing_source_penalty"]) * missing_count,
            min_score,
            max_score,
        ),
        1,
    )
    rule_ai_score = round(
        clamp(
            float(scoring_cfg["bank_weight_transcript"]) * rule_transcript_mean
            + float(scoring_cfg["bank_weight_10k"]) * rule_tenk_mean
            - float(scoring_cfg["missing_source_penalty"]) * missing_count,
            min_score,
            max_score,
        ),
        1,
    )
    llm_ai_score = round(
        clamp(
            float(scoring_cfg["bank_weight_transcript"]) * llm_transcript_mean
            + float(scoring_cfg["bank_weight_10k"]) * llm_tenk_mean
            - float(scoring_cfg["missing_source_penalty"]) * missing_count,
            min_score,
            max_score,
        ),
        1,
    )

    all_scores = transcript_scores + tenk_scores
    if all_scores:
        best = max(
            all_scores,
            key=lambda d: (d.final_score, d.signal_count, 1 if d.doc_type == "transcript" else 0),
        )
        evidence = best.evidence_summary
        source = f"{best.doc_type}:{best.period}"
    else:
        evidence = "No source documents were available for this bank."
        source = "none"

    signal_count = sum(d.signal_count for d in all_scores)
    llm_confidence = round(safe_mean([d.confidence for d in all_scores], 0.0), 2)

    return BankOutputRow(
        Bank=bank_name,
        Ticker=ticker,
        AI_Score=ai_score,
        Rule_AI_Score=rule_ai_score,
        LLM_AI_Score=llm_ai_score,
        AI_Score_Normalized=1.0,
        Evidence=evidence,
        Evidence_Source=source,
        signal_count=signal_count,
        llm_confidence=llm_confidence,
        missing_sources="none" if not missing else ",".join(missing),
        num_transcripts=len(transcript_scores),
        num_10k_docs=len(tenk_scores),
    )


def recompute_hybrid_ai_score(row: BankOutputRow, config: dict[str, Any]) -> float:
    scoring_cfg = config["scoring"]
    min_score = float(scoring_cfg["min_score"])
    max_score = float(scoring_cfg["max_score"])
    score = (
        float(scoring_cfg["doc_weight_rule"]) * row.Rule_AI_Score
        + float(scoring_cfg["doc_weight_llm"]) * row.LLM_AI_Score
    )
    return round(clamp(score, min_score, max_score), 1)


def calibrate_llm_scores(rows: list[BankOutputRow], config: dict[str, Any], *, verbose: bool = False) -> None:
    if not rows:
        return

    scoring_cfg = config["scoring"]
    if not bool(scoring_cfg.get("llm_calibration_enabled", True)):
        for row in rows:
            row.AI_Score = recompute_hybrid_ai_score(row, config)
        return

    llm_scores = [row.LLM_AI_Score for row in rows]
    rule_scores = [row.Rule_AI_Score for row in rows]
    if len(llm_scores) < 5:
        for row in rows:
            row.AI_Score = recompute_hybrid_ai_score(row, config)
        return

    llm_span = max(llm_scores) - min(llm_scores)
    llm_std = statistics.pstdev(llm_scores)
    span_threshold = float(scoring_cfg.get("llm_compression_span_threshold", 2.0))
    std_threshold = float(scoring_cfg.get("llm_compression_std_threshold", 0.55))
    is_compressed = llm_span <= span_threshold or llm_std <= std_threshold
    if not is_compressed:
        for row in rows:
            row.AI_Score = recompute_hybrid_ai_score(row, config)
        return

    min_score = float(scoring_cfg["min_score"])
    max_score = float(scoring_cfg["max_score"])
    llm_mean = statistics.fmean(llm_scores)
    rule_mean = statistics.fmean(rule_scores)
    rule_std = statistics.pstdev(rule_scores)
    target_std = max(
        float(scoring_cfg.get("llm_target_std_floor", 0.9)),
        float(scoring_cfg.get("llm_target_std_factor", 0.8)) * rule_std,
    )
    rule_blend = clamp(float(scoring_cfg.get("llm_calibration_rule_blend", 0.3)), 0.0, 1.0)
    max_hybrid_gap = max(
        0.0,
        float(scoring_cfg.get("max_hybrid_gap_below_rule_when_llm_compressed", 0.5)),
    )

    if verbose:
        print(
            "Calibrating compressed LLM scores: "
            f"span={llm_span:.2f}, std={llm_std:.2f}, target_std={target_std:.2f}",
            file=sys.stderr,
        )

    for row in rows:
        if llm_std < 1e-6:
            mapped = rule_mean
        else:
            z_value = (row.LLM_AI_Score - llm_mean) / llm_std
            mapped = rule_mean + z_value * target_std
        calibrated = (1.0 - rule_blend) * mapped + rule_blend * row.Rule_AI_Score
        row.LLM_AI_Score = round(clamp(calibrated, min_score, max_score), 1)
        hybrid = recompute_hybrid_ai_score(row, config)
        floor_score = round(clamp(row.Rule_AI_Score - max_hybrid_gap, min_score, max_score), 1)
        row.AI_Score = max(hybrid, floor_score)


def apply_percentile_normalization(rows: list[BankOutputRow]) -> None:
    if not rows:
        return
    sorted_scores = sorted(row.AI_Score for row in rows)
    n = len(sorted_scores)

    rank_map: dict[float, float] = {}
    for score in sorted(set(sorted_scores)):
        positions = [idx + 1 for idx, value in enumerate(sorted_scores) if value == score]
        average_rank = statistics.fmean(positions)
        if n == 1:
            percentile = 0.5
        else:
            percentile = (average_rank - 1.0) / (n - 1.0)
        rank_map[score] = percentile

    for row in rows:
        percentile = rank_map[row.AI_Score]
        row.AI_Score_Normalized = round(1.0 + 9.0 * percentile, 1)


def write_csv(rows: list[BankOutputRow], output_path: Path) -> None:
    if not rows:
        return
    headers = list(asdict(rows[0]).keys())
    with output_path.open("w", encoding="utf-8", newline="") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def write_xlsx(rows: list[BankOutputRow], output_path: Path) -> None:
    if not rows:
        return
    headers = list(asdict(rows[0]).keys())
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "AI_Bank_Classification"
    sheet.append(headers)
    for row in rows:
        sheet.append([asdict(row)[header] for header in headers])
    workbook.save(output_path)


def run_pipeline(
    *,
    sec_zip_path: Path,
    transcript_zip_path: Path,
    config_path: Path | None,
    backend: LLMBackend | None,
    out_csv: Path | None,
    out_xlsx: Path | None,
    verbose: bool,
    max_banks: int | None,
) -> list[BankOutputRow]:
    config = load_config(config_path)

    with ZipFile(transcript_zip_path) as transcript_zip, ZipFile(sec_zip_path) as sec_zip:
        transcript_docs = index_transcript_docs(transcript_zip)
        tenk_docs, full_by_ticker = index_sec_docs(sec_zip)
        bank_names = derive_bank_names(sec_zip, full_by_ticker)

        tickers = sorted(set(transcript_docs.keys()) | set(tenk_docs.keys()))
        if max_banks is not None:
            tickers = tickers[:max_banks]

        if verbose:
            print(f"Discovered {len(tickers)} tickers for scoring.", file=sys.stderr)

        rows: list[BankOutputRow] = []

        for index, ticker in enumerate(tickers, start=1):
            t_docs = transcript_docs.get(ticker, [])
            k_docs = tenk_docs.get(ticker, [])
            bank_name = bank_names.get(ticker, ticker)

            transcript_scores: list[DocumentScore] = []
            for doc in t_docs:
                doc.bank_name = bank_name
                raw_text = transcript_zip.read(doc.zip_member_path).decode("utf-8", errors="ignore")
                cleaned = clean_transcript_text(raw_text)
                transcript_scores.append(score_document(doc, cleaned, backend, config, verbose))

            tenk_scores: list[DocumentScore] = []
            for doc in k_docs:
                doc.bank_name = bank_name
                raw_html = sec_zip.read(doc.zip_member_path).decode("utf-8", errors="ignore")
                cleaned = clean_10k_html(raw_html)
                tenk_scores.append(score_document(doc, cleaned, backend, config, verbose))

            row = aggregate_bank(ticker, bank_name, transcript_scores, tenk_scores, config)
            rows.append(row)

            if verbose and index % 5 == 0:
                print(f"Scored {index}/{len(tickers)} tickers...", file=sys.stderr)

    calibrate_llm_scores(rows, config, verbose=verbose)
    apply_percentile_normalization(rows)
    rows.sort(key=lambda r: r.Ticker)

    if out_csv:
        write_csv(rows, out_csv)
    if out_xlsx:
        write_xlsx(rows, out_xlsx)

    return rows


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Classify AI focus for banks using transcript and 10-K zip data.")
    parser.add_argument("--sec-zip", type=Path, required=True, help="Path to SEC filings zip archive")
    parser.add_argument("--transcript-zip", type=Path, required=True, help="Path to transcripts zip archive")
    parser.add_argument(
        "--provider",
        choices=["huggingface", "gemini", "none"],
        default="huggingface",
        help="LLM provider for hybrid scoring",
    )
    parser.add_argument(
        "--hf-model",
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Hugging Face model id",
    )
    parser.add_argument(
        "--gemini-model",
        default="gemini-2.5-pro",
        help="Gemini model id (scaffold placeholder)",
    )
    parser.add_argument("--hf-token-env", default="HF_TOKEN", help="Environment variable containing HF token")
    parser.add_argument("--hf-timeout", type=int, default=90, help="HF API timeout in seconds")
    parser.add_argument("--hf-retries", type=int, default=3, help="HF API retry count")
    parser.add_argument(
        "--hf-local",
        action="store_true",
        help="Use local Hugging Face Transformers inference instead of router API",
    )
    parser.add_argument(
        "--hf-local-device",
        choices=["auto", "cpu", "cuda", "mps"],
        default="auto",
        help="Device for local Hugging Face inference",
    )
    parser.add_argument(
        "--hf-local-max-new-tokens",
        type=int,
        default=220,
        help="Maximum tokens to generate per local model response",
    )
    parser.add_argument(
        "--hf-local-max-input-tokens",
        type=int,
        default=2048,
        help="Maximum prompt tokens passed to local model",
    )
    parser.add_argument(
        "--allow-rule-fallback",
        action="store_true",
        help="Allow running without live LLM calls if provider credentials/backend are unavailable",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT_DIR / "config" / "scoring.yml",
        help="Path to scoring config yaml",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=ROOT_DIR / "AI_Bank_Classification.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--out-xlsx",
        type=Path,
        default=ROOT_DIR / "AI_Bank_Classification.xlsx",
        help="Output Excel path",
    )
    parser.add_argument("--max-banks", type=int, default=None, help="Optional limit for debug runs")
    parser.add_argument("--verbose", action="store_true", help="Enable progress logging")
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    if not args.sec_zip.exists():
        raise SystemExit(f"SEC zip not found: {args.sec_zip}")
    if not args.transcript_zip.exists():
        raise SystemExit(f"Transcript zip not found: {args.transcript_zip}")

    backend: LLMBackend | None
    try:
        backend = choose_backend(args)
    except BackendError as exc:
        if args.allow_rule_fallback:
            print(f"Backend unavailable ({exc}); using rule-only fallback.", file=sys.stderr)
            backend = None
        else:
            raise

    rows = run_pipeline(
        sec_zip_path=args.sec_zip,
        transcript_zip_path=args.transcript_zip,
        config_path=args.config,
        backend=backend,
        out_csv=args.out_csv,
        out_xlsx=args.out_xlsx,
        verbose=args.verbose,
        max_banks=args.max_banks,
    )

    print(f"Scored {len(rows)} banks.")
    print(f"CSV: {args.out_csv}")
    print(f"XLSX: {args.out_xlsx}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
