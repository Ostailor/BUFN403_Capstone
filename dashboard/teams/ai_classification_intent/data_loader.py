"""
Shared data-loading helpers for the AI Classification & Intent team plugin.

Each loader tries to read the canonical dashboard artifact first
(`bank_composite_scores.csv`, `quarterly_progression.csv`,
`app_category_matrix.csv`, `classifications.jsonl`). If a derived CSV is
missing but `classifications.jsonl` exists, the loader rebuilds the
dashboard view from those Qwen-backed classifications. If neither exists,
it returns an empty frame and lets the page show a clean missing-data
state instead of silently fabricating numbers.
"""

from pathlib import Path

import pandas as pd
import streamlit as st

from dashboard.core.paths import team_artifacts_dir
from src.ai_corpus.classification_io import CLASSIFICATION_COLUMNS, read_classifications_jsonl
from src.ai_corpus.composite_scorer import (
    APP_CATEGORY_COLUMNS,
    QUARTERLY_COLUMNS,
    SCORES_COLUMNS,
    build_dashboard_rows,
)


# ── Helpers ─────────────────────────────────────────────────────────

def _empty_frame(columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=columns)


def _read_canonical_csv(path: Path, columns: list[str]) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return _empty_frame(columns)
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return _empty_frame(columns)


def _dashboard_rows_from_classifications(data_dir: Path) -> tuple[list[dict], list[dict], list[dict]]:
    classifications_path = data_dir / "classifications.jsonl"
    if not classifications_path.exists():
        return [], [], []
    return build_dashboard_rows(read_classifications_jsonl(classifications_path))


# ── Composite Scores ────────────────────────────────────────────────

@st.cache_data
def load_scores() -> pd.DataFrame:
    """Load bank_composite_scores.csv or rebuild it from classifications.jsonl."""
    data_dir = team_artifacts_dir(__file__)

    canonical = data_dir / "bank_composite_scores.csv"
    if canonical.exists():
        return _read_canonical_csv(canonical, SCORES_COLUMNS)

    composite_rows, _, _ = _dashboard_rows_from_classifications(data_dir)
    if composite_rows:
        return pd.DataFrame(composite_rows, columns=SCORES_COLUMNS)

    return _empty_frame(SCORES_COLUMNS)


# ── Quarterly Progression ───────────────────────────────────────────

@st.cache_data
def load_quarterly() -> pd.DataFrame:
    """Load quarterly_progression.csv or rebuild it from classifications.jsonl."""
    data_dir = team_artifacts_dir(__file__)

    canonical = data_dir / "quarterly_progression.csv"
    if canonical.exists():
        return _read_canonical_csv(canonical, QUARTERLY_COLUMNS)

    _, quarterly_rows, _ = _dashboard_rows_from_classifications(data_dir)
    if quarterly_rows:
        return pd.DataFrame(quarterly_rows, columns=QUARTERLY_COLUMNS)

    return _empty_frame(QUARTERLY_COLUMNS)


# ── Application-Category Matrix ─────────────────────────────────────

@st.cache_data
def load_app_categories() -> pd.DataFrame:
    """Load app_category_matrix.csv or rebuild it from classifications.jsonl."""
    data_dir = team_artifacts_dir(__file__)

    canonical = data_dir / "app_category_matrix.csv"
    if canonical.exists():
        return _read_canonical_csv(canonical, APP_CATEGORY_COLUMNS)

    _, _, category_rows = _dashboard_rows_from_classifications(data_dir)
    if category_rows:
        return pd.DataFrame(category_rows, columns=APP_CATEGORY_COLUMNS)

    return _empty_frame(APP_CATEGORY_COLUMNS)


# ── Classifications (JSONL) ─────────────────────────────────────────

@st.cache_data
def load_classifications() -> pd.DataFrame:
    """Load classifications.jsonl with normalized list-valued categories."""
    data_dir = team_artifacts_dir(__file__)

    canonical = data_dir / "classifications.jsonl"
    if canonical.exists():
        return pd.DataFrame(read_classifications_jsonl(canonical), columns=CLASSIFICATION_COLUMNS)

    return _empty_frame(CLASSIFICATION_COLUMNS)
