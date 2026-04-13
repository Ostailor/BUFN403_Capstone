"""
Shared data-loading helpers for the BUFN403 AI-Intent Dashboard.

Each loader tries to read the canonical dashboard CSV first
(artifacts/ai_corpus/bank_composite_scores.csv, etc.).  If that file
doesn't exist, it falls back to *deriving* the data from the raw
pipeline artifacts that DO exist (ai_bank_scorecard.csv,
topic_findings.csv, ai_theme_examples.csv).  If neither source is
available it generates deterministic mock data so the dashboard is
always renderable.
"""

import json
import pandas as pd
import numpy as np
import streamlit as st
from pathlib import Path

# ── Constants ────────────────────────────────────────────────────────

MOCK_BANKS = [
    ("JPM", "JPMorgan Chase"),
    ("BAC", "Bank of America"),
    ("WFC", "Wells Fargo"),
    ("C", "Citigroup"),
    ("GS", "Goldman Sachs"),
    ("MS", "Morgan Stanley"),
    ("USB", "US Bancorp"),
    ("PNC", "PNC Financial"),
]

THEME_CATEGORIES = [
    "customer_facing_ai", "operations_efficiency", "risk_controls",
    "use_cases", "investment_spend", "measurable_outcomes",
    "vendors_partnerships", "governance", "ai_strategy",
]

THEME_LABELS = {
    "customer_facing_ai": "Customer-Facing AI",
    "operations_efficiency": "Operations Efficiency",
    "risk_controls": "Risk Controls",
    "use_cases": "Use Cases",
    "investment_spend": "Investment & Spend",
    "measurable_outcomes": "Measurable Outcomes",
    "vendors_partnerships": "Vendors & Partnerships",
    "governance": "Governance",
    "ai_strategy": "AI Strategy",
}


def _data_dir(reference_file: str) -> Path:
    """Return the artifacts/ai_corpus directory relative to *reference_file*.

    Works correctly whether called from dashboard/app.py (parents[1]) or
    dashboard/pages/*.py (parents[2]).
    """
    p = Path(reference_file).resolve()
    # Walk up until we find the project root (contains 'artifacts/')
    for ancestor in p.parents:
        candidate = ancestor / "artifacts" / "ai_corpus"
        if candidate.is_dir():
            return candidate
    # Fallback: assume two levels up from pages, one from dashboard
    if p.parent.name == "pages":
        return p.resolve().parents[2] / "artifacts" / "ai_corpus"
    return p.resolve().parents[1] / "artifacts" / "ai_corpus"


# ── Composite Scores ────────────────────────────────────────────────

@st.cache_data
def load_scores(reference_file: str) -> pd.DataFrame:
    """Load bank_composite_scores.csv or derive from ai_bank_scorecard.csv."""
    data_dir = _data_dir(reference_file)

    # 1) Canonical file
    canonical = data_dir / "bank_composite_scores.csv"
    if canonical.exists():
        return pd.read_csv(canonical)

    # 2) Derive from scorecard
    scorecard = data_dir / "ai_bank_scorecard.csv"
    if scorecard.exists():
        sc = pd.read_csv(scorecard)
        # Execution themes → Maturity (normalised 0-100)
        exec_cols = [c for c in ["customer_facing_ai", "operations_efficiency",
                                  "use_cases", "measurable_outcomes",
                                  "investment_spend", "vendors_partnerships"]
                     if c in sc.columns]
        gov_cols = [c for c in ["governance", "risk_controls", "ai_strategy"]
                    if c in sc.columns]

        sc["_exec_raw"] = sc[exec_cols].sum(axis=1) if exec_cols else 0
        sc["_gov_raw"] = sc[gov_cols].sum(axis=1) if gov_cols else 0
        sc["_total_raw"] = sc["_exec_raw"] + sc["_gov_raw"]

        max_total = sc["_total_raw"].max() or 1
        max_exec = sc["_exec_raw"].max() or 1
        max_gov = sc["_gov_raw"].max() or 1

        sc["Maturity"] = (sc["_total_raw"] / max_total * 100).round(1)
        sc["Breadth"] = (sc[exec_cols + gov_cols].gt(0).sum(axis=1)
                         / len(exec_cols + gov_cols) * 100).round(1) if (exec_cols + gov_cols) else 0.0

        # Momentum: governance_strategy_share as a proxy (higher = more strategic)
        if "governance_strategy_share" in sc.columns:
            sc["Momentum"] = (sc["governance_strategy_share"] * 100 - 20).round(1)
        else:
            sc["Momentum"] = 0.0

        sc["Composite"] = (0.5 * sc["Maturity"] + 0.3 * sc["Breadth"]
                           + 0.2 * sc["Momentum"]).round(1)
        sc = sc.sort_values("Composite", ascending=False).reset_index(drop=True)
        sc["Rank"] = sc.index + 1

        out = sc.rename(columns={"ticker": "Ticker", "bank_name": "Bank"})[
            ["Ticker", "Bank", "Maturity", "Breadth", "Momentum", "Composite", "Rank"]
        ].copy()
        return out

    # 3) Mock
    return _mock_scores()


def _mock_scores() -> pd.DataFrame:
    rng = np.random.RandomState(42)
    rows = []
    for ticker, name in MOCK_BANKS:
        maturity = round(rng.uniform(30, 95), 1)
        breadth = round(rng.uniform(20, 90), 1)
        momentum = round(rng.uniform(-10, 30), 1)
        composite = round(0.5 * maturity + 0.3 * breadth + 0.2 * momentum, 1)
        rows.append(dict(Ticker=ticker, Bank=name, Maturity=maturity,
                         Breadth=breadth, Momentum=momentum, Composite=composite, Rank=0))
    df = pd.DataFrame(rows).sort_values("Composite", ascending=False).reset_index(drop=True)
    df["Rank"] = df.index + 1
    return df


# ── Quarterly Progression ───────────────────────────────────────────

@st.cache_data
def load_quarterly(reference_file: str) -> pd.DataFrame:
    """Load quarterly_progression.csv or derive from topic_findings.csv."""
    data_dir = _data_dir(reference_file)

    canonical = data_dir / "quarterly_progression.csv"
    if canonical.exists():
        return pd.read_csv(canonical)

    # Derive from topic_findings.csv
    findings = data_dir / "topic_findings.csv"
    if findings.exists():
        tf = pd.read_csv(findings)
        # Assign each theme_tag to an intent bucket
        exploring_tags = {"ai_strategy", "governance"}
        committing_tags = {"investment_spend", "vendors_partnerships"}
        deploying_tags = {"use_cases", "operations_efficiency", "risk_controls"}
        scaling_tags = {"customer_facing_ai", "measurable_outcomes"}

        def _bucket(tag):
            if tag in exploring_tags:
                return "Exploring"
            if tag in committing_tags:
                return "Committing"
            if tag in deploying_tags:
                return "Deploying"
            return "Scaling"

        tf["bucket"] = tf["theme_tag"].apply(_bucket)
        grp = tf.groupby(["ticker", "period_year", "period_quarter", "bucket"])[
            "mention_count"].sum().reset_index()
        total = grp.groupby(["ticker", "period_year", "period_quarter"])[
            "mention_count"].transform("sum")
        grp["pct"] = (grp["mention_count"] / total * 100).round(1)
        pivot = grp.pivot_table(index=["ticker", "period_year", "period_quarter"],
                                columns="bucket", values="pct", fill_value=0).reset_index()
        pivot.columns.name = None
        for col in ["Exploring", "Committing", "Deploying", "Scaling"]:
            if col not in pivot.columns:
                pivot[col] = 0.0

        # Maturity score: weighted sum where Scaling=4, Deploying=3, etc.
        pivot["Maturity_Score"] = (
            1 * pivot["Exploring"] + 2 * pivot["Committing"]
            + 3 * pivot["Deploying"] + 4 * pivot["Scaling"]
        ).round(1) / 4  # normalise

        pivot = pivot.rename(columns={
            "ticker": "Ticker",
            "period_year": "Year",
            "period_quarter": "Quarter",
            "Exploring": "Exploring_Pct",
            "Committing": "Committing_Pct",
            "Deploying": "Deploying_Pct",
            "Scaling": "Scaling_Pct",
        })
        return pivot[["Ticker", "Year", "Quarter", "Maturity_Score",
                       "Exploring_Pct", "Committing_Pct", "Deploying_Pct",
                       "Scaling_Pct"]].copy()

    return _mock_quarterly()


def _mock_quarterly() -> pd.DataFrame:
    rng = np.random.RandomState(77)
    rows = []
    for ticker, _ in MOCK_BANKS:
        for year in [2023, 2024, 2025]:
            for q in [1, 2, 3, 4]:
                if year == 2025 and q > 1:
                    continue
                exploring = round(rng.uniform(10, 40), 1)
                committing = round(rng.uniform(15, 35), 1)
                deploying = round(rng.uniform(10, 30), 1)
                scaling = max(round(100 - exploring - committing - deploying, 1), 0)
                maturity_score = round(rng.uniform(30, 90) + (year - 2023) * 5 + q * 1.5, 1)
                rows.append(dict(Ticker=ticker, Year=year, Quarter=q,
                                 Maturity_Score=maturity_score,
                                 Exploring_Pct=exploring, Committing_Pct=committing,
                                 Deploying_Pct=deploying, Scaling_Pct=scaling))
    return pd.DataFrame(rows)


# ── Application-Category Matrix ─────────────────────────────────────

@st.cache_data
def load_app_categories(reference_file: str) -> pd.DataFrame:
    """Load app_category_matrix.csv or derive from topic_findings.csv / scorecard."""
    data_dir = _data_dir(reference_file)

    canonical = data_dir / "app_category_matrix.csv"
    if canonical.exists():
        return pd.read_csv(canonical)

    # Derive from topic_findings (per-bank per-theme totals)
    findings = data_dir / "topic_findings.csv"
    if findings.exists():
        tf = pd.read_csv(findings)
        grp = tf.groupby(["ticker", "theme_tag"])["mention_count"].sum().reset_index()
        grp = grp.rename(columns={
            "ticker": "Ticker",
            "theme_tag": "Category",
            "mention_count": "Mention_Count",
        })
        # Map theme_tag to readable label
        grp["Category"] = grp["Category"].map(THEME_LABELS).fillna(grp["Category"])
        # Approximate intent level as normalised mention count 1-4
        max_m = grp["Mention_Count"].max() or 1
        grp["Avg_Intent_Level"] = (grp["Mention_Count"] / max_m * 3 + 1).round(2)
        return grp[["Ticker", "Category", "Mention_Count", "Avg_Intent_Level"]].copy()

    # Derive from scorecard (wide → long)
    scorecard = data_dir / "ai_bank_scorecard.csv"
    if scorecard.exists():
        sc = pd.read_csv(scorecard)
        theme_cols = [c for c in THEME_CATEGORIES if c in sc.columns]
        melted = sc.melt(id_vars=["ticker"], value_vars=theme_cols,
                         var_name="Category", value_name="Mention_Count")
        melted = melted.rename(columns={"ticker": "Ticker"})
        melted["Category"] = melted["Category"].map(THEME_LABELS).fillna(melted["Category"])
        melted["Mention_Count"] = melted["Mention_Count"].fillna(0).astype(int)
        max_m = melted["Mention_Count"].max() or 1
        melted["Avg_Intent_Level"] = (melted["Mention_Count"] / max_m * 3 + 1).round(2)
        return melted[["Ticker", "Category", "Mention_Count", "Avg_Intent_Level"]].copy()

    return _mock_app_categories()


def _mock_app_categories() -> pd.DataFrame:
    rng = np.random.RandomState(99)
    categories = ["Risk Management", "Customer Service", "Fraud Detection",
                   "Trading", "Compliance", "Lending"]
    rows = []
    for ticker, _ in MOCK_BANKS:
        for cat in categories:
            rows.append(dict(Ticker=ticker, Category=cat,
                             Mention_Count=int(rng.randint(1, 50)),
                             Avg_Intent_Level=round(rng.uniform(1, 4), 2)))
    return pd.DataFrame(rows)


# ── Classifications (JSONL) ─────────────────────────────────────────

@st.cache_data
def load_classifications(reference_file: str) -> pd.DataFrame:
    """Load classifications.jsonl or derive from ai_theme_examples.csv."""
    data_dir = _data_dir(reference_file)

    canonical = data_dir / "classifications.jsonl"
    if canonical.exists():
        records = []
        with open(canonical) as f:
            for line in f:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    # Ensure app_categories is always a list
                    cats = rec.get("app_categories", [])
                    if isinstance(cats, str):
                        try:
                            cats = json.loads(cats)
                        except (json.JSONDecodeError, TypeError):
                            cats = [c.strip() for c in cats.split(",") if c.strip()]
                    rec["app_categories"] = cats if isinstance(cats, list) else []
                    records.append(rec)
        return pd.DataFrame(records)

    # Derive from ai_theme_examples.csv
    examples = data_dir / "ai_theme_examples.csv"
    if examples.exists():
        ex = pd.read_csv(examples)
        rows = []
        for _, r in ex.iterrows():
            label = THEME_LABELS.get(r.get("theme_tag", ""), r.get("theme_label", ""))
            intent_map = {
                "ai_strategy": ("Exploring", 1),
                "governance": ("Exploring", 1),
                "investment_spend": ("Committing", 2),
                "vendors_partnerships": ("Committing", 2),
                "use_cases": ("Deploying", 3),
                "operations_efficiency": ("Deploying", 3),
                "risk_controls": ("Deploying", 3),
                "customer_facing_ai": ("Scaling", 4),
                "measurable_outcomes": ("Scaling", 4),
            }
            intent_label, intent_level = intent_map.get(
                r.get("theme_tag", ""), ("Exploring", 1))
            rows.append(dict(
                chunk_id=r.get("chunk_id", ""),
                ticker=r.get("ticker", ""),
                bank_name=r.get("bank_name", ""),
                source_type=r.get("source_type", ""),
                period_year=int(r["period_year"]) if pd.notna(r.get("period_year")) else 0,
                period_quarter=int(r["period_quarter"]) if pd.notna(r.get("period_quarter")) else 0,
                intent_level=intent_level,
                intent_label=intent_label,
                app_categories=[label],
                confidence=0.8,
                evidence_snippet=str(r.get("snippet", ""))[:300],
            ))
        return pd.DataFrame(rows)

    return _mock_classifications()


def _mock_classifications() -> pd.DataFrame:
    rng = np.random.RandomState(55)
    intent_labels = ["Exploring", "Committing", "Deploying", "Scaling"]
    source_types = ["earnings_call", "10-K", "press_release"]
    categories = ["Risk Management", "Customer Service", "Fraud Detection",
                   "Trading", "Compliance", "Lending"]
    rows = []
    for ticker, name in MOCK_BANKS:
        for i in range(15):
            rows.append(dict(
                chunk_id=f"{ticker}_{i:03d}",
                ticker=ticker,
                bank_name=name,
                source_type=rng.choice(source_types),
                period_year=int(rng.choice([2023, 2024, 2025])),
                period_quarter=int(rng.choice([1, 2, 3, 4])),
                intent_level=int(rng.choice([1, 2, 3, 4])),
                intent_label=rng.choice(intent_labels),
                app_categories=list(rng.choice(categories,
                                               size=rng.randint(1, 4), replace=False)),
                confidence=round(rng.uniform(0.6, 1.0), 2),
                evidence_snippet=f"Sample evidence snippet for {name} chunk {i}."
            ))
    return pd.DataFrame(rows)
