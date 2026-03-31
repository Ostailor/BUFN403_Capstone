#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
import zipfile
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
AI_CORPUS_DIR = ROOT / "artifacts" / "ai_corpus"
FFIEC_DIR = ROOT / "artifacts" / "ffiec_bulk"
OUTPUT_DIR = ROOT / "artifacts" / "april_1_ai_team"
ROSTER_CSV = ROOT / "AI_Bank_Classification.csv"

COMMON_PERIODS = [
    (2024, 1),
    (2024, 2),
    (2024, 3),
    (2024, 4),
    (2025, 1),
    (2025, 2),
    (2025, 3),
    (2025, 4),
]

FFIEC_REQUIRED_FILES = [
    "FFIEC-CDR-Call-Bulk-All-Schedules-03312024.zip",
    "FFIEC-CDR-Call-Bulk-All-Schedules-06302024.zip",
    "FFIEC-CDR-Call-Bulk-All-Schedules-09302024.zip",
    "FFIEC-CDR-Call-Bulk-All-Schedules-12312024.zip",
    "FFIEC-CDR-Call-Bulk-All-Schedules-03312025.zip",
    "FFIEC-CDR-Call-Bulk-All-Schedules-06302025.zip",
    "FFIEC-CDR-Call-Bulk-All-Schedules-09302025.zip",
    "FFIEC-CDR-Call-Bulk-All-Schedules-12312025.zip",
]

FDIC_CERT_MAP = {
    "ALLY": 57803,
    "ASB": 5296,
    "AXP": 27471,
    "BAC": 3510,
    "BK": 639,
    "BKU": 58979,
    "BOKF": 4214,
    "BPOP": 34968,
    "C": 7213,
    "CFG": 57957,
    "CFR": 5510,
    "CMA": 983,
    "COF": 4297,
    "COLB": 17266,
    "DFS": 5649,
    "EWBC": 31628,
    "FCNCA": 11063,
    "FHN": 4977,
    "FITB": 6672,
    "FLG": 32541,
    "FNB": 7888,
    "GS": 33124,
    "HBAN": 6560,
    "JPM": 628,
    "KEY": 17534,
    "MS": 32992,
    "MTB": 588,
    "NTRS": 913,
    "ONB": 3832,
    "PB": 16835,
    "PNC": 6384,
    "PNFP": 35583,
    "RF": 12368,
    "RJF": 33893,
    "SCHW": 57450,
    "SF": 57311,
    "SNV": 873,
    "SOFI": 26881,
    "SSB": 33555,
    "STT": 14,
    "SYF": 27314,
    "TFC": 9846,
    "UMBF": 8273,
    "USB": 6548,
    "VLY": 9396,
    "WAL": 57512,
    "WBS": 18221,
    "WFC": 3511,
    "WTFC": 33935,
    "ZION": 2270,
}

AI_GOVERNANCE_THEMES = ["ai_strategy", "governance", "risk_controls"]
AI_EXECUTION_THEMES = [
    "customer_facing_ai",
    "investment_spend",
    "measurable_outcomes",
    "operations_efficiency",
    "use_cases",
    "vendors_partnerships",
]

PRIVATE_CREDIT_PATTERNS: dict[str, tuple[re.Pattern[str], float]] = {
    "private_credit": (re.compile(r"\bprivate credit\b", re.I), 1.0),
    "direct_lending": (re.compile(r"\bdirect lending\b|\bdirect lenders?\b", re.I), 1.0),
    "private_debt": (re.compile(r"\bprivate debt\b", re.I), 1.0),
    "sponsor_finance": (re.compile(r"\bsponsor finance\b", re.I), 1.0),
    "fund_finance": (
        re.compile(r"\bfund finance\b|\bsubscription lines?\b|\bcapital call lines?\b", re.I),
        0.9,
    ),
    "asset_based_lending": (re.compile(r"\basset[- ]based lending\b", re.I), 0.6),
    "private_capital": (re.compile(r"\bprivate capital\b", re.I), 0.6),
}

FDIC_FINANCIAL_FIELDS = [
    "CERT",
    "REPDTE",
    "NAME",
    "ASSET",
    "DEP",
    "NETINC",
    "ROA",
    "ROE",
    "EQ",
    "NIMY",
    "NCLNLS",
    "NCLNLSR",
    "P3ASSET",
    "P9ASSET",
    "COREDEP",
    "DEPDOM",
    "DEPINS",
    "DEPUNINS",
    "LNLSNET",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build April 1 AI team deliverables.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory where the April 1 deliverables will be written.",
    )
    return parser.parse_args()


def period_label(year: int, quarter: int) -> str:
    return f"{year}_Q{quarter}"


def month_to_quarter(month: int) -> int:
    return ((month - 1) // 3) + 1


def numeric_frame(frame: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    frame = frame.copy()
    for column in columns:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    numerator = pd.to_numeric(numerator, errors="coerce")
    denominator = pd.to_numeric(denominator, errors="coerce")
    result = numerator / denominator
    result[denominator <= 0] = np.nan
    return result


def sparse_percentile(values: pd.Series) -> pd.Series:
    values = pd.to_numeric(values, errors="coerce").fillna(0.0)
    result = pd.Series(0.0, index=values.index)
    positive = values > 0
    if positive.any():
        result.loc[positive] = values.loc[positive].rank(method="average", pct=True) * 100.0
    return result


def percentile(values: pd.Series, higher_is_better: bool = True) -> pd.Series:
    values = pd.to_numeric(values, errors="coerce")
    result = pd.Series(np.nan, index=values.index)
    valid = values.notna()
    if valid.any():
        result.loc[valid] = values.loc[valid].rank(
            method="average",
            pct=True,
            ascending=higher_is_better,
        ) * 100.0
    return result


def rating_from_score(values: pd.Series) -> pd.Series:
    values = pd.to_numeric(values, errors="coerce")
    bins = [-np.inf, 20, 40, 60, 80, np.inf]
    labels = [1, 2, 3, 4, 5]
    rated = pd.cut(values, bins=bins, labels=labels)
    return rated.astype("float").astype("Int64")


def weighted_row_score(frame: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    weighted_sum = pd.Series(0.0, index=frame.index)
    weight_sum = pd.Series(0.0, index=frame.index)
    for column, weight in weights.items():
        if column not in frame.columns:
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        mask = values.notna()
        weighted_sum.loc[mask] += values.loc[mask] * weight
        weight_sum.loc[mask] += weight
    score = weighted_sum / weight_sum.replace(0, np.nan)
    return score


def recent_summary(frame: pd.DataFrame, prefix: str, value_column: str, breadth_column: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (ticker, bank_name), group in frame.groupby(["ticker", "bank_name"], sort=False):
        recent = group.sort_values(["period_year", "period_quarter"]).tail(4)
        latest = recent.iloc[-1]
        first = recent.iloc[0]
        rows.append(
            {
                "ticker": ticker,
                "bank_name": bank_name,
                f"{prefix}_current_period_label": latest["period_label"],
                f"{prefix}_latest_value": latest[value_column],
                f"{prefix}_trailing_4q_value": recent[value_column].sum(),
                f"{prefix}_avg_breadth": recent[breadth_column].mean(),
                f"{prefix}_momentum": latest[value_column] - first[value_column],
            }
        )
    return pd.DataFrame(rows)


def full_period_grid(roster: pd.DataFrame, periods: list[tuple[int, int]]) -> pd.DataFrame:
    rows = []
    for item in roster.itertuples(index=False):
        for year, quarter in periods:
            rows.append(
                {
                    "ticker": item.Ticker,
                    "bank_name": item.Bank,
                    "period_year": year,
                    "period_quarter": quarter,
                    "period_label": period_label(year, quarter),
                }
            )
    return pd.DataFrame(rows)


def load_roster() -> pd.DataFrame:
    roster = pd.read_csv(ROSTER_CSV, usecols=["Ticker", "Bank"])
    roster["Ticker"] = roster["Ticker"].str.upper()
    return roster


def load_fdic_cert_mapping(roster: pd.DataFrame) -> pd.DataFrame:
    session = requests.Session()
    institution_rows: list[dict[str, object]] = []
    offset = 0
    limit = 10000
    while True:
        response = session.get(
            "https://api.fdic.gov/banks/institutions",
            params={
                "format": "json",
                "limit": limit,
                "offset": offset,
                "fields": "NAME,CERT,NAMEHCR,CITY,STALP,ACTIVE,ASSET",
            },
            timeout=60,
        )
        response.raise_for_status()
        payload = response.json()
        institution_rows.extend(item["data"] for item in payload.get("data", []))
        offset += limit
        if offset >= int(payload.get("meta", {}).get("total", 0)):
            break
    institutions = pd.DataFrame(institution_rows)
    mapping = roster.copy()
    mapping["cert"] = mapping["Ticker"].map(FDIC_CERT_MAP)
    mapping = mapping.merge(institutions, left_on="cert", right_on="CERT", how="left")
    mapping = mapping.rename(
        columns={
            "Ticker": "ticker",
            "Bank": "bank_name",
            "NAME": "institution_name",
            "NAMEHCR": "holding_company",
            "CITY": "city",
            "STALP": "state",
            "ACTIVE": "active_flag",
            "ASSET": "fdic_institution_assets",
        }
    )
    return mapping[
        [
            "ticker",
            "bank_name",
            "cert",
            "institution_name",
            "holding_company",
            "city",
            "state",
            "active_flag",
            "fdic_institution_assets",
        ]
    ].sort_values("ticker")


def build_ai_quarterly(roster: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    topic = pd.read_csv(AI_CORPUS_DIR / "topic_findings.csv")
    topic["ticker"] = topic["ticker"].str.upper()
    topic["period_label"] = topic.apply(
        lambda row: period_label(int(row["period_year"]), int(row["period_quarter"])),
        axis=1,
    )
    topic = topic[
        topic["period_label"].isin({period_label(year, quarter) for year, quarter in COMMON_PERIODS})
    ].copy()

    pivot = topic.pivot_table(
        index=["ticker", "period_year", "period_quarter", "period_label"],
        columns="theme_tag",
        values="mention_count",
        aggfunc="sum",
        fill_value=0,
    ).reset_index()
    for theme in AI_GOVERNANCE_THEMES + AI_EXECUTION_THEMES:
        if theme not in pivot.columns:
            pivot[theme] = 0

    source_breadth = (
        topic.groupby(["ticker", "period_year", "period_quarter", "period_label"], as_index=False)[
            "source_type"
        ]
        .nunique()
        .rename(columns={"source_type": "source_breadth"})
    )

    grid = full_period_grid(roster, COMMON_PERIODS)
    ai = (
        grid.merge(pivot, on=["ticker", "period_year", "period_quarter", "period_label"], how="left")
        .merge(source_breadth, on=["ticker", "period_year", "period_quarter", "period_label"], how="left")
        .fillna(0)
    )

    theme_columns = [column for column in ai.columns if column in AI_GOVERNANCE_THEMES + AI_EXECUTION_THEMES]
    ai[theme_columns + ["source_breadth"]] = ai[theme_columns + ["source_breadth"]].apply(
        pd.to_numeric, errors="coerce"
    ).fillna(0)
    ai["total_mentions"] = ai[theme_columns].sum(axis=1)
    ai["governance_mentions"] = ai[AI_GOVERNANCE_THEMES].sum(axis=1)
    ai["execution_mentions"] = ai[AI_EXECUTION_THEMES].sum(axis=1)
    ai["execution_share"] = safe_divide(ai["execution_mentions"], ai["total_mentions"]).fillna(0.0)

    ai["quarterly_score"] = np.nan
    for _, idx in ai.groupby("period_label").groups.items():
        ai.loc[idx, "quarterly_score"] = (
            0.65 * sparse_percentile(ai.loc[idx, "total_mentions"])
            + 0.20 * sparse_percentile(ai.loc[idx, "source_breadth"])
            + 0.15 * sparse_percentile(ai.loc[idx, "execution_mentions"])
        )
    ai["quarterly_rating"] = rating_from_score(ai["quarterly_score"])

    current = recent_summary(ai, "ai", "total_mentions", "source_breadth")
    execution_recent = (
        ai.sort_values(["period_year", "period_quarter"])
        .groupby(["ticker", "bank_name"], as_index=False)
        .tail(4)
        .groupby(["ticker", "bank_name"], as_index=False)
        .agg(ai_avg_execution_share=("execution_share", "mean"))
    )
    current = current.merge(execution_recent, on=["ticker", "bank_name"], how="left")
    current["ai_current_score"] = weighted_row_score(
        pd.DataFrame(
            {
                "trailing": sparse_percentile(current["ai_trailing_4q_value"]),
                "latest": sparse_percentile(current["ai_latest_value"]),
                "breadth": sparse_percentile(current["ai_avg_breadth"]),
                "execution": percentile(current["ai_avg_execution_share"], higher_is_better=True).fillna(0),
                "momentum": percentile(current["ai_momentum"], higher_is_better=True).fillna(0),
            }
        ),
        {
            "trailing": 0.40,
            "latest": 0.25,
            "breadth": 0.15,
            "execution": 0.10,
            "momentum": 0.10,
        },
    )
    current["ai_current_rating"] = rating_from_score(current["ai_current_score"])
    return ai, current


def build_private_credit_quarterly(roster: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    hits: list[dict[str, object]] = []
    chunks_path = AI_CORPUS_DIR / "chunks.jsonl"
    with chunks_path.open("r", encoding="utf-8") as infile:
        for line in infile:
            chunk = json.loads(line)
            period = (int(chunk["period_year"]), int(chunk["period_quarter"]))
            if period not in COMMON_PERIODS:
                continue
            text = str(chunk["chunk_text"])
            for family, (pattern, weight) in PRIVATE_CREDIT_PATTERNS.items():
                if pattern.search(text):
                    hits.append(
                        {
                            "ticker": chunk["ticker"].upper(),
                            "bank_name": chunk["bank_name"],
                            "period_year": int(chunk["period_year"]),
                            "period_quarter": int(chunk["period_quarter"]),
                            "period_label": period_label(int(chunk["period_year"]), int(chunk["period_quarter"])),
                            "source_type": chunk["source_type"],
                            "doc_id": chunk["doc_id"],
                            "chunk_id": chunk["chunk_id"],
                            "family": family,
                            "weight": weight,
                            "example_text": text[:500].replace("\n", " ").strip(),
                        }
                    )
    hits_df = pd.DataFrame(hits)
    if hits_df.empty:
        raise RuntimeError("No private-credit hits were found in artifacts/ai_corpus/chunks.jsonl.")

    hit_summary = (
        hits_df.groupby(["ticker", "period_year", "period_quarter", "period_label"], as_index=False)
        .agg(
            weighted_hits=("weight", "sum"),
            matched_chunks=("chunk_id", "nunique"),
            matched_docs=("doc_id", "nunique"),
            family_breadth=("family", "nunique"),
            source_breadth=("source_type", "nunique"),
        )
        .sort_values(["ticker", "period_year", "period_quarter"])
    )
    family_pivot = hits_df.pivot_table(
        index=["ticker", "period_year", "period_quarter", "period_label"],
        columns="family",
        values="weight",
        aggfunc="sum",
        fill_value=0.0,
    ).reset_index()
    grid = full_period_grid(roster, COMMON_PERIODS)
    private_credit = (
        grid.merge(
            hit_summary,
            on=["ticker", "period_year", "period_quarter", "period_label"],
            how="left",
        )
        .merge(
            family_pivot,
            on=["ticker", "period_year", "period_quarter", "period_label"],
            how="left",
        )
        .fillna(0)
    )
    family_columns = [column for column in private_credit.columns if column in PRIVATE_CREDIT_PATTERNS]
    private_credit[["weighted_hits", "matched_chunks", "matched_docs", "family_breadth", "source_breadth"] + family_columns] = private_credit[
        ["weighted_hits", "matched_chunks", "matched_docs", "family_breadth", "source_breadth"] + family_columns
    ].apply(pd.to_numeric, errors="coerce").fillna(0)

    private_credit["quarterly_score"] = np.nan
    for _, idx in private_credit.groupby("period_label").groups.items():
        private_credit.loc[idx, "quarterly_score"] = (
            0.55 * sparse_percentile(private_credit.loc[idx, "weighted_hits"])
            + 0.25 * sparse_percentile(private_credit.loc[idx, "family_breadth"])
            + 0.20 * sparse_percentile(private_credit.loc[idx, "source_breadth"])
        )
    private_credit["quarterly_rating"] = rating_from_score(private_credit["quarterly_score"])

    current = recent_summary(private_credit, "private_credit", "weighted_hits", "family_breadth")
    current = current.merge(
        (
            private_credit.sort_values(["period_year", "period_quarter"])
            .groupby(["ticker", "bank_name"], as_index=False)
            .tail(4)
            .groupby(["ticker", "bank_name"], as_index=False)
            .agg(private_credit_avg_source_breadth=("source_breadth", "mean"))
        ),
        on=["ticker", "bank_name"],
        how="left",
    )
    current["private_credit_current_score"] = weighted_row_score(
        pd.DataFrame(
            {
                "trailing": sparse_percentile(current["private_credit_trailing_4q_value"]),
                "latest": sparse_percentile(current["private_credit_latest_value"]),
                "breadth": sparse_percentile(current["private_credit_avg_breadth"]),
                "source_breadth": sparse_percentile(current["private_credit_avg_source_breadth"]),
                "momentum": percentile(current["private_credit_momentum"], higher_is_better=True).fillna(0),
            }
        ),
        {
            "trailing": 0.40,
            "latest": 0.25,
            "breadth": 0.15,
            "source_breadth": 0.10,
            "momentum": 0.10,
        },
    )
    current["private_credit_current_rating"] = rating_from_score(current["private_credit_current_score"])
    return hits_df, private_credit, current


def fetch_fdic_financials(cert_map: pd.DataFrame) -> pd.DataFrame:
    session = requests.Session()
    rows: list[dict[str, object]] = []
    for item in cert_map.itertuples(index=False):
        response = session.get(
            "https://api.fdic.gov/banks/financials",
            params={
                "format": "json",
                "limit": 200,
                "sort_by": "REPDTE",
                "sort_order": "DESC",
                "fields": ",".join(FDIC_FINANCIAL_FIELDS),
                "filters": f"CERT:{item.cert}",
            },
            timeout=60,
        )
        response.raise_for_status()
        payload = response.json()
        for result in payload.get("data", []):
            data = result.get("data", {})
            repdte = str(data.get("REPDTE", ""))
            if len(repdte) != 8 or not repdte.isdigit():
                continue
            year = int(repdte[:4])
            quarter = month_to_quarter(int(repdte[4:6]))
            if (year, quarter) not in COMMON_PERIODS:
                continue
            row = {
                "ticker": item.ticker,
                "bank_name": item.bank_name,
                "cert": item.cert,
                "period_year": year,
                "period_quarter": quarter,
                "period_label": period_label(year, quarter),
                "risk_report_date": repdte,
            }
            for field in FDIC_FINANCIAL_FIELDS:
                row[field] = data.get(field)
            rows.append(row)
    fdic = pd.DataFrame(rows)
    fdic = numeric_frame(
        fdic,
        [
            "ASSET",
            "DEP",
            "NETINC",
            "ROA",
            "ROE",
            "EQ",
            "NIMY",
            "NCLNLS",
            "NCLNLSR",
            "P3ASSET",
            "P9ASSET",
            "COREDEP",
            "DEPDOM",
            "DEPINS",
            "DEPUNINS",
            "LNLSNET",
        ],
    )
    fdic["equity_assets_ratio"] = safe_divide(fdic["EQ"], fdic["ASSET"]) * 100.0
    fdic["deposit_assets_ratio"] = safe_divide(fdic["DEP"], fdic["ASSET"]) * 100.0
    fdic["core_deposit_ratio"] = safe_divide(fdic["COREDEP"], fdic["DEP"]) * 100.0
    return fdic.sort_values(["ticker", "period_year", "period_quarter"])


def _coalesce(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    available = [column for column in columns if column in frame.columns]
    if not available:
        return pd.Series(np.nan, index=frame.index)
    values = frame[available].apply(pd.to_numeric, errors="coerce")
    return values.bfill(axis=1).iloc[:, 0]


def build_ffiec_dpd(cert_map: pd.DataFrame) -> pd.DataFrame:
    missing = [filename for filename in FFIEC_REQUIRED_FILES if not (FFIEC_DIR / filename).exists()]
    if missing:
        raise FileNotFoundError(
            "Missing FFIEC bulk files required for DPD calculations: " + ", ".join(missing)
        )

    cert_lookup = cert_map[["ticker", "bank_name", "cert"]].copy()
    wanted_certs = {int(cert) for cert in cert_lookup["cert"].tolist()}
    rows: list[pd.DataFrame] = []

    for zip_path in sorted(FFIEC_DIR.glob("FFIEC-CDR-Call-Bulk-All-Schedules-*.zip")):
        mmddyyyy = zip_path.stem.split("-")[-1]
        year = int(mmddyyyy[-4:])
        month = int(mmddyyyy[:2])
        quarter = month_to_quarter(month)
        if (year, quarter) not in COMMON_PERIODS:
            continue
        with zipfile.ZipFile(zip_path) as archive:
            por_name = f"FFIEC CDR Call Bulk POR {mmddyyyy}.txt"
            rcci_name = f"FFIEC CDR Call Schedule RCCI {mmddyyyy}.txt"
            rcn_name = f"FFIEC CDR Call Schedule RCN {mmddyyyy}.txt"

            por = pd.read_csv(
                archive.open(por_name),
                sep="\t",
                dtype=str,
                usecols=["IDRSSD", "FDIC Certificate Number", "Financial Institution Name"],
            )
            por = por.rename(
                columns={
                    "FDIC Certificate Number": "cert",
                    "Financial Institution Name": "ffiec_institution_name",
                }
            )
            por["cert"] = pd.to_numeric(por["cert"], errors="coerce")
            por = por[por["cert"].isin(wanted_certs)].copy()

            rcci = pd.read_csv(
                archive.open(rcci_name),
                sep="\t",
                dtype=str,
                usecols=lambda column: column
                in {
                    "IDRSSD",
                    "RCFD1763",
                    "RCFD1764",
                    "RCON1763",
                    "RCON1764",
                    "RCFDJ454",
                    "RCONJ454",
                },
            )
            rcn = pd.read_csv(
                archive.open(rcn_name),
                sep="\t",
                dtype=str,
                usecols=lambda column: column
                in {
                    "IDRSSD",
                    "RCFD1252",
                    "RCFD1253",
                    "RCFD1255",
                    "RCFD1256",
                    "RCON1252",
                    "RCON1253",
                    "RCON1255",
                    "RCON1256",
                    "RCON1607",
                    "RCFDPV25",
                    "RCONPV25",
                },
            )

            merged = por.merge(rcci, on="IDRSSD", how="left").merge(rcn, on="IDRSSD", how="left")
            merged = merged.merge(cert_lookup, on="cert", how="left")
            merged["period_year"] = year
            merged["period_quarter"] = quarter
            merged["period_label"] = period_label(year, quarter)

            ci_loan_us_90 = _coalesce(merged, ["RCFD1252", "RCON1607"]).fillna(0.0)
            ci_loan_us_nonaccrual = _coalesce(merged, ["RCFD1253", "RCON1253"]).fillna(0.0)
            ci_loan_nonus_90 = _coalesce(merged, ["RCFD1255", "RCON1255"]).fillna(0.0)
            ci_loan_nonus_nonaccrual = _coalesce(merged, ["RCFD1256", "RCON1256"]).fillna(0.0)
            ci_loan_denominator = _coalesce(merged, ["RCFD1763", "RCON1763"]).fillna(0.0) + _coalesce(
                merged, ["RCFD1764", "RCON1764"]
            ).fillna(0.0)
            ci_loan_numerator = (
                ci_loan_us_90 + ci_loan_us_nonaccrual + ci_loan_nonus_90 + ci_loan_nonus_nonaccrual
            )
            requested_nondepository_numerator = _coalesce(merged, ["RCFDPV25", "RCONPV25"])
            requested_nondepository_denominator = _coalesce(merged, ["RCFDJ454", "RCONJ454"])

            merged["commercial_loan_90_plus_dpd_numerator"] = ci_loan_numerator
            merged["commercial_loan_90_plus_dpd_denominator"] = ci_loan_denominator
            merged["commercial_loan_90_plus_dpd_rate"] = safe_divide(ci_loan_numerator, ci_loan_denominator)
            merged["nondepository_requested_numerator"] = requested_nondepository_numerator
            merged["nondepository_requested_denominator"] = requested_nondepository_denominator
            merged["nondepository_requested_rate"] = safe_divide(
                requested_nondepository_numerator, requested_nondepository_denominator
            )

            rows.append(
                merged[
                    [
                        "ticker",
                        "bank_name",
                        "cert",
                        "ffiec_institution_name",
                        "period_year",
                        "period_quarter",
                        "period_label",
                        "commercial_loan_90_plus_dpd_numerator",
                        "commercial_loan_90_plus_dpd_denominator",
                        "commercial_loan_90_plus_dpd_rate",
                        "nondepository_requested_numerator",
                        "nondepository_requested_denominator",
                        "nondepository_requested_rate",
                    ]
                ]
            )
    ffiec = pd.concat(rows, ignore_index=True)
    return ffiec.sort_values(["ticker", "period_year", "period_quarter"])


def build_risk_quarterly(roster: pd.DataFrame, fdic: pd.DataFrame, ffiec: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    grid = full_period_grid(roster, COMMON_PERIODS)
    risk = grid.merge(
        fdic,
        on=["ticker", "bank_name", "period_year", "period_quarter", "period_label"],
        how="left",
    ).merge(
        ffiec,
        on=["ticker", "bank_name", "period_year", "period_quarter", "period_label"],
        how="left",
    )

    metric_weights = {
        "roa_score": 0.20,
        "roe_score": 0.15,
        "nimy_score": 0.10,
        "equity_assets_score": 0.15,
        "noncurrent_loans_score": 0.15,
        "commercial_dpd_score": 0.15,
        "nondepository_requested_score": 0.10,
    }

    risk["quarterly_score"] = np.nan
    for _, idx in risk.groupby("period_label").groups.items():
        quarter = risk.loc[idx].copy()
        quarter["roa_score"] = percentile(quarter["ROA"], higher_is_better=True)
        quarter["roe_score"] = percentile(quarter["ROE"], higher_is_better=True)
        quarter["nimy_score"] = percentile(quarter["NIMY"], higher_is_better=True)
        quarter["equity_assets_score"] = percentile(quarter["equity_assets_ratio"], higher_is_better=True)
        quarter["noncurrent_loans_score"] = percentile(quarter["NCLNLSR"], higher_is_better=False)
        quarter["commercial_dpd_score"] = percentile(
            quarter["commercial_loan_90_plus_dpd_rate"], higher_is_better=False
        )
        quarter["nondepository_requested_score"] = percentile(
            quarter["nondepository_requested_rate"], higher_is_better=False
        )
        risk.loc[idx, "quarterly_score"] = weighted_row_score(quarter, metric_weights)
    risk["quarterly_rating"] = rating_from_score(risk["quarterly_score"])

    latest_rows: list[pd.Series] = []
    for _, group in risk.groupby(["ticker", "bank_name"], sort=False):
        available = group[group["quarterly_score"].notna()].sort_values(["period_year", "period_quarter"])
        if available.empty:
            latest_rows.append(group.sort_values(["period_year", "period_quarter"]).iloc[-1])
        else:
            latest_rows.append(available.iloc[-1])
    current = pd.DataFrame(latest_rows)
    current = current.rename(
        columns={
            "period_label": "risk_current_period_label",
            "quarterly_score": "risk_current_score",
            "quarterly_rating": "risk_current_rating",
        }
    )
    return risk, current[
        [
            "ticker",
            "bank_name",
            "risk_current_period_label",
            "risk_current_score",
            "risk_current_rating",
            "ROA",
            "ROE",
            "NIMY",
            "equity_assets_ratio",
            "NCLNLSR",
            "commercial_loan_90_plus_dpd_rate",
            "nondepository_requested_rate",
            "ASSET",
        ]
    ]


def assign_clusters(frame: pd.DataFrame, feature_columns: list[str], labels: list[str], score_column: str) -> pd.DataFrame:
    model_input = frame[feature_columns].apply(pd.to_numeric, errors="coerce")
    model_input = model_input.fillna(model_input.median(numeric_only=True)).fillna(0.0)
    std = model_input.std(ddof=0).replace(0, 1.0)
    scaled = (model_input - model_input.mean()) / std
    cluster_index = simple_kmeans(scaled.to_numpy(dtype=float), n_clusters=3, random_state=42, n_init=50)
    result = frame[["ticker", "bank_name", score_column]].copy()
    result["cluster_id"] = cluster_index
    ordering = (
        result.groupby("cluster_id", as_index=False)[score_column]
        .mean()
        .sort_values(score_column, ascending=False)
        .reset_index(drop=True)
    )
    label_map = {
        int(row["cluster_id"]): labels[position]
        for position, row in ordering.iterrows()
    }
    result["cluster_label"] = result["cluster_id"].map(label_map)
    return result.drop(columns=["cluster_id"])


def simple_kmeans(
    values: np.ndarray,
    n_clusters: int,
    random_state: int = 42,
    n_init: int = 25,
    max_iter: int = 200,
) -> np.ndarray:
    rng = np.random.default_rng(random_state)
    best_labels: np.ndarray | None = None
    best_inertia = math.inf
    if len(values) < n_clusters:
        raise ValueError("Not enough rows to assign clusters.")

    for _ in range(n_init):
        centers = values[rng.choice(len(values), size=n_clusters, replace=False)].copy()
        labels = np.zeros(len(values), dtype=int)
        for _ in range(max_iter):
            distances = ((values[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
            new_labels = distances.argmin(axis=1)
            new_centers = centers.copy()
            for cluster_id in range(n_clusters):
                cluster_mask = new_labels == cluster_id
                if cluster_mask.any():
                    new_centers[cluster_id] = values[cluster_mask].mean(axis=0)
                else:
                    new_centers[cluster_id] = values[rng.integers(0, len(values))]
            if np.array_equal(new_labels, labels) and np.allclose(new_centers, centers):
                labels = new_labels
                centers = new_centers
                break
            labels = new_labels
            centers = new_centers
        inertia = ((values - centers[labels]) ** 2).sum()
        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels.copy()

    if best_labels is None:
        raise RuntimeError("K-means clustering did not converge to a valid solution.")
    return best_labels


def build_driver_correlations(
    ai_quarterly: pd.DataFrame,
    ai_current: pd.DataFrame,
    private_credit_hits: pd.DataFrame,
    private_credit_current: pd.DataFrame,
    risk_current: pd.DataFrame,
) -> pd.DataFrame:
    drivers: list[dict[str, object]] = []

    ai_recent = ai_quarterly.sort_values(["period_year", "period_quarter"]).groupby(["ticker", "bank_name"]).tail(4)
    ai_theme_recent = (
        ai_recent.groupby(["ticker", "bank_name"], as_index=False)[AI_GOVERNANCE_THEMES + AI_EXECUTION_THEMES]
        .sum()
        .merge(ai_current[["ticker", "bank_name", "ai_current_score"]], on=["ticker", "bank_name"], how="left")
    )
    for column in AI_GOVERNANCE_THEMES + AI_EXECUTION_THEMES:
        if ai_theme_recent[column].std(ddof=0) == 0:
            continue
        drivers.append(
            {
                "workstream": "ai_activity",
                "feature": column,
                "correlation": ai_theme_recent[column].corr(ai_theme_recent["ai_current_score"]),
            }
        )

    private_recent = (
        private_credit_hits.groupby(["ticker", "bank_name", "family"], as_index=False)["weight"]
        .sum()
        .pivot_table(index=["ticker", "bank_name"], columns="family", values="weight", fill_value=0.0)
        .reset_index()
        .merge(
            private_credit_current[["ticker", "bank_name", "private_credit_current_score"]],
            on=["ticker", "bank_name"],
            how="left",
        )
    )
    for column in PRIVATE_CREDIT_PATTERNS:
        if column not in private_recent.columns or private_recent[column].std(ddof=0) == 0:
            continue
        drivers.append(
            {
                "workstream": "private_credit",
                "feature": column,
                "correlation": private_recent[column].corr(private_recent["private_credit_current_score"]),
            }
        )

    risk_features = [
        "ROA",
        "ROE",
        "NIMY",
        "equity_assets_ratio",
        "NCLNLSR",
        "commercial_loan_90_plus_dpd_rate",
        "nondepository_requested_rate",
    ]
    risk_driver_frame = risk_current[["ticker", "bank_name", "risk_current_score"] + risk_features].copy()
    for column in risk_features:
        if risk_driver_frame[column].std(ddof=0) == 0:
            continue
        drivers.append(
            {
                "workstream": "risk_resilience",
                "feature": column,
                "correlation": risk_driver_frame[column].corr(risk_driver_frame["risk_current_score"]),
            }
        )

    return pd.DataFrame(drivers).sort_values(["workstream", "correlation"], ascending=[True, False])


def build_manual_ai_priority(
    current_ratings: pd.DataFrame,
    risk_current: pd.DataFrame,
) -> pd.DataFrame:
    manifest = pd.read_csv(AI_CORPUS_DIR / "document_manifest.csv")
    manifest = manifest[
        manifest["period_year"].fillna(0).between(2024, 2025)
        & manifest["source_type"].isin(["transcript", "sec_filing", "call_report"])
    ].copy()

    summary = (
        manifest.groupby(["ticker"], as_index=False)
        .agg(
            missing_docs=("status", lambda values: int(sum(value == "missing" for value in values))),
            partial_docs=("status", lambda values: int(sum(value == "partial" for value in values))),
            not_public_docs=("status", lambda values: int(sum(value == "not_public" for value in values))),
            manual_search_hint=("manual_search_hint", lambda values: " | ".join(sorted({str(v) for v in values if pd.notna(v) and str(v).strip()}))[:800]),
        )
    )
    priority = current_ratings.merge(summary, on="ticker", how="left")
    for column in ["missing_docs", "partial_docs", "not_public_docs"]:
        priority[column] = priority[column].fillna(0).astype(int)
    priority["manual_search_priority_score"] = (
        0.50 * sparse_percentile(priority["missing_docs"] + priority["partial_docs"])
        + 0.25 * percentile(priority["ai_current_score"], higher_is_better=True).fillna(0)
        + 0.25 * percentile(priority["private_credit_current_score"], higher_is_better=True).fillna(0)
    )
    priority = priority.sort_values(
        ["manual_search_priority_score", "missing_docs", "partial_docs", "ai_current_score"],
        ascending=[False, False, False, False],
    )
    return priority[
        [
            "ticker",
            "bank_name",
            "manual_search_priority_score",
            "missing_docs",
            "partial_docs",
            "not_public_docs",
            "ai_current_rating",
            "private_credit_current_rating",
            "risk_current_period_label",
            "manual_search_hint",
        ]
    ]


def write_manual_ai_files(
    output_dir: Path,
    priority: pd.DataFrame,
    generated_files: list[Path],
) -> tuple[Path, Path]:
    upload_manifest = output_dir / "manual_ai_upload_manifest.csv"
    upload_rows = [
        {"file_path": str(path), "purpose": purpose}
        for path, purpose in [
            (output_dir / "april_1_ai_team_summary.md", "Executive summary and caveats"),
            (output_dir / "workstream_current_ratings.csv", "Current ratings across the three workstreams"),
            (output_dir / "workstream_quarterly_ratings.csv", "Quarterly ratings across the three workstreams"),
            (output_dir / "workstream_clusters.csv", "Cluster assignments for AI, private credit, and risk"),
            (output_dir / "workstream_driver_correlations.csv", "Driver correlations by workstream"),
            (output_dir / "manual_ai_priority_banks.csv", "Banks to prioritize for manual web search"),
            (output_dir / "risk_resilience_index_detail.xlsx", "Transparent workbook for the risk index and DPD metrics"),
            (output_dir / "private_credit_hits.csv", "Underlying private-credit hit extraction"),
            (AI_CORPUS_DIR / "document_manifest.csv", "Document coverage and search hints"),
            (AI_CORPUS_DIR / "topic_findings.csv", "Underlying AI topic mentions"),
            (AI_CORPUS_DIR / "ai_bank_scorecard.csv", "Existing AI scorecard baseline"),
        ]
        if path.exists()
    ]
    pd.DataFrame(upload_rows).to_csv(upload_manifest, index=False)

    priority_lines = "\n".join(
        f"- {row.ticker}: {row.bank_name}" for row in priority.head(12).itertuples(index=False)
    )
    prompt_path = output_dir / "manual_ai_gemini_chatgpt_prompt.md"
    prompt_path.write_text(
        "\n".join(
            [
                "# Manual Gemini / ChatGPT Search Prompt",
                "",
                "Upload the files listed in `manual_ai_upload_manifest.csv`, then use this prompt:",
                "",
                "```text",
                "You are helping extend a bank-capstone analysis that already has local ratings and clustering for three workstreams:",
                "1. AI activity",
                "2. private credit activity",
                "3. risk resilience",
                "",
                "Use the uploaded files as the current local baseline. Then search beyond the uploaded local corpus and identify only incremental evidence that materially changes or sharpens the conclusions.",
                "",
                "Rules:",
                "- Prioritize official bank sources, investor presentations, earnings-call transcripts, SEC filings, and reputable financial reporting.",
                "- Focus first on the banks below because local coverage is thinner or mixed there.",
                "- For each bank, flag whether new external evidence changes the AI, private-credit, or risk-resilience view.",
                "- Give exact source links and publication dates.",
                "- Do not restate what is already in the uploaded files unless it is needed to explain a delta.",
                "- If no incremental value exists for a bank, say so briefly.",
                "",
                "Priority banks:",
                priority_lines,
                "",
                "Output format:",
                "1. Bank",
                "2. Workstream affected",
                "3. New evidence summary",
                "4. Why it changes or does not change the local rating",
                "5. Source link",
                "```",
                "",
                "Note: the local workbook includes the requested commercial-loan 90+ DPD formula. The requested nondepository formula was also computed as provided, but it should be field-validated before any final presentation claims.",
            ]
        ),
        encoding="utf-8",
    )
    return upload_manifest, prompt_path


def write_summary(
    output_dir: Path,
    current_ratings: pd.DataFrame,
    clusters: pd.DataFrame,
    drivers: pd.DataFrame,
) -> Path:
    summary_path = output_dir / "april_1_ai_team_summary.md"

    def top_table(score_column: str, rating_column: str) -> str:
        top = current_ratings.sort_values(score_column, ascending=False).head(10)
        return "\n".join(
            f"- {row.ticker}: {row.bank_name} (score {getattr(row, score_column):.1f}, rating {int(getattr(row, rating_column)) if pd.notna(getattr(row, rating_column)) else 'NA'})"
            for row in top.itertuples(index=False)
        )

    driver_lines = []
    for workstream, group in drivers.groupby("workstream"):
        top = group.sort_values("correlation", ascending=False).head(5)
        driver_lines.append(f"## {workstream.replace('_', ' ').title()}")
        for row in top.itertuples(index=False):
            if pd.isna(row.correlation):
                continue
            driver_lines.append(f"- {row.feature}: correlation {row.correlation:.2f}")
        driver_lines.append("")

    cluster_counts = (
        clusters.groupby(["workstream", "cluster"])
        .size()
        .reset_index(name="bank_count")
        .sort_values(["workstream", "bank_count"], ascending=[True, False])
    )
    cluster_lines = [
        f"- {row.workstream}: {row.cluster} ({row.bank_count})"
        for row in cluster_counts.itertuples(index=False)
    ]

    summary_path.write_text(
        "\n".join(
            [
                "# April 1 AI Team Deliverables",
                "",
                "This package turns the current repo into a working April 1 deliverable set for the three inferred focus areas:",
                "- AI activity",
                "- private credit activity",
                "- risk resilience / credit quality",
                "",
                "## What Was Generated",
                "",
                "- Current bank-level ratings across the three workstreams",
                "- Quarterly ratings for 2024_Q1 through 2025_Q4",
                "- Cluster assignments for each workstream",
                "- A transparent risk workbook with the Risk Resilience Index inputs and FFIEC-based DPD metrics",
                "- A manual Gemini / ChatGPT packet for the external-search task that still sits outside the local pipeline",
                "",
                "## Top AI Activity Banks",
                top_table("ai_current_score", "ai_current_rating"),
                "",
                "## Top Private Credit Banks",
                top_table("private_credit_current_score", "private_credit_current_rating"),
                "",
                "## Top Risk Resilience Banks",
                top_table("risk_current_score", "risk_current_rating"),
                "",
                "## Cluster Counts",
                *cluster_lines,
                "",
                "## Driver Snapshots",
                *driver_lines,
                "## Key Caveats",
                "",
                "- The three workstreams are inferred from the project context and meeting notes; they are not explicitly codified elsewhere in the repo.",
                "- The commercial-loan 90+ DPD metric follows the requested FFIEC call-report formula with `RCFD/RCON` fallback for banks where one branch is blank.",
                "- The requested nondepository formula was computed exactly as provided (`RCFDPV25 / RCFDJ454`, with `RCON` fallback), but the FFIEC field labels suggest the numerator should be validated before making a final presentation claim that it represents a full nondepository 90+ DPD rate.",
                "- A few banks have latest risk data on a slightly different as-of quarter in the FDIC API; the current risk sheet includes the exact quarter used per bank.",
            ]
        ),
        encoding="utf-8",
    )
    return summary_path


def main() -> None:
    args = parse_args()
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    roster = load_roster()
    cert_map = load_fdic_cert_mapping(roster)
    cert_map.to_csv(output_dir / "fdic_cert_mapping.csv", index=False)

    ai_quarterly, ai_current = build_ai_quarterly(roster)
    private_credit_hits, private_credit_quarterly, private_credit_current = build_private_credit_quarterly(roster)
    fdic_financials = fetch_fdic_financials(cert_map)
    ffiec_dpd = build_ffiec_dpd(cert_map)
    risk_quarterly, risk_current = build_risk_quarterly(roster, fdic_financials, ffiec_dpd)

    current_ratings = (
        roster.rename(columns={"Ticker": "ticker", "Bank": "bank_name"})
        .merge(ai_current, on=["ticker", "bank_name"], how="left")
        .merge(private_credit_current, on=["ticker", "bank_name"], how="left")
        .merge(risk_current, on=["ticker", "bank_name"], how="left")
    )

    ai_clusters = assign_clusters(
        ai_current,
        [
            "ai_current_score",
            "ai_trailing_4q_value",
            "ai_latest_value",
            "ai_avg_breadth",
            "ai_avg_execution_share",
        ],
        ["AI Leaders", "AI Active", "AI Watchlist"],
        "ai_current_score",
    ).rename(columns={"cluster_label": "cluster", "ai_current_score": "current_score"})
    ai_clusters["workstream"] = "ai_activity"
    ai_clusters["rating"] = current_ratings.set_index("ticker").loc[ai_clusters["ticker"], "ai_current_rating"].values
    ai_clusters = ai_clusters[["ticker", "bank_name", "workstream", "cluster", "rating", "current_score"]]

    private_credit_clusters = assign_clusters(
        private_credit_current,
        [
            "private_credit_current_score",
            "private_credit_trailing_4q_value",
            "private_credit_latest_value",
            "private_credit_avg_breadth",
            "private_credit_avg_source_breadth",
        ],
        ["Private Credit Leaders", "Private Credit Active", "Private Credit Limited"],
        "private_credit_current_score",
    ).rename(columns={"cluster_label": "cluster", "private_credit_current_score": "current_score"})
    private_credit_clusters["workstream"] = "private_credit"
    private_credit_clusters["rating"] = current_ratings.set_index("ticker").loc[
        private_credit_clusters["ticker"], "private_credit_current_rating"
    ].values
    private_credit_clusters = private_credit_clusters[
        ["ticker", "bank_name", "workstream", "cluster", "rating", "current_score"]
    ]

    risk_cluster_features = risk_current.copy()
    risk_clusters = assign_clusters(
        risk_cluster_features,
        [
            "risk_current_score",
            "ROA",
            "equity_assets_ratio",
            "NCLNLSR",
            "commercial_loan_90_plus_dpd_rate",
            "nondepository_requested_rate",
        ],
        ["Risk Resilient", "Risk Moderate", "Risk Watch"],
        "risk_current_score",
    ).rename(columns={"cluster_label": "cluster", "risk_current_score": "current_score"})
    risk_clusters["workstream"] = "risk_resilience"
    risk_clusters["rating"] = current_ratings.set_index("ticker").loc[risk_clusters["ticker"], "risk_current_rating"].values
    risk_clusters = risk_clusters[["ticker", "bank_name", "workstream", "cluster", "rating", "current_score"]]

    clusters = pd.concat([ai_clusters, private_credit_clusters, risk_clusters], ignore_index=True)

    quarterly_ratings = pd.concat(
        [
            ai_quarterly[
                ["ticker", "bank_name", "period_year", "period_quarter", "period_label", "quarterly_score", "quarterly_rating"]
            ].assign(workstream="ai_activity"),
            private_credit_quarterly[
                ["ticker", "bank_name", "period_year", "period_quarter", "period_label", "quarterly_score", "quarterly_rating"]
            ].assign(workstream="private_credit"),
            risk_quarterly[
                ["ticker", "bank_name", "period_year", "period_quarter", "period_label", "quarterly_score", "quarterly_rating"]
            ].assign(workstream="risk_resilience"),
        ],
        ignore_index=True,
    ).rename(columns={"quarterly_score": "score", "quarterly_rating": "rating"})

    drivers = build_driver_correlations(
        ai_quarterly,
        ai_current,
        private_credit_hits,
        private_credit_current,
        risk_current,
    )

    priority = build_manual_ai_priority(current_ratings, risk_current)

    current_ratings.to_csv(output_dir / "workstream_current_ratings.csv", index=False)
    quarterly_ratings.to_csv(output_dir / "workstream_quarterly_ratings.csv", index=False)
    clusters.to_csv(output_dir / "workstream_clusters.csv", index=False)
    drivers.to_csv(output_dir / "workstream_driver_correlations.csv", index=False)
    ai_quarterly.to_csv(output_dir / "ai_quarterly_scores.csv", index=False)
    private_credit_quarterly.to_csv(output_dir / "private_credit_quarterly_scores.csv", index=False)
    risk_quarterly.to_csv(output_dir / "risk_quarterly_scores.csv", index=False)
    private_credit_hits.to_csv(output_dir / "private_credit_hits.csv", index=False)
    fdic_financials.to_csv(output_dir / "fdic_quarterly_financials.csv", index=False)
    ffiec_dpd.to_csv(output_dir / "ffiec_dpd_metrics.csv", index=False)
    priority.to_csv(output_dir / "manual_ai_priority_banks.csv", index=False)

    workbook_path = output_dir / "risk_resilience_index_detail.xlsx"
    with pd.ExcelWriter(workbook_path, engine="openpyxl") as writer:
        current_ratings.to_excel(writer, index=False, sheet_name="current_ratings")
        quarterly_ratings.to_excel(writer, index=False, sheet_name="quarterly_ratings")
        risk_quarterly.to_excel(writer, index=False, sheet_name="risk_quarterly")
        ffiec_dpd.to_excel(writer, index=False, sheet_name="ffiec_dpd_metrics")
        fdic_financials.to_excel(writer, index=False, sheet_name="fdic_financials")
        clusters.to_excel(writer, index=False, sheet_name="clusters")
        drivers.to_excel(writer, index=False, sheet_name="drivers")
        priority.to_excel(writer, index=False, sheet_name="manual_ai_priority")

    summary_path = write_summary(output_dir, current_ratings, clusters, drivers)
    write_manual_ai_files(
        output_dir,
        priority,
        [
            output_dir / "workstream_current_ratings.csv",
            output_dir / "workstream_quarterly_ratings.csv",
            output_dir / "workstream_clusters.csv",
            workbook_path,
            summary_path,
        ],
    )


if __name__ == "__main__":
    main()
