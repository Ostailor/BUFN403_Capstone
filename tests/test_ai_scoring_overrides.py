"""Tests for ``dashboard.teams.ai_classification_intent.scoring`` overrides."""

from __future__ import annotations

import pandas as pd
import pytest

from dashboard.teams.ai_classification_intent.scoring import (
    DEFAULT_WEIGHTS,
    format_weights_caption,
    normalize_weights,
    recompute_scores,
)


def _canonical_composite(maturity: float, breadth: float, momentum: float,
                          weights: dict[str, float]) -> float:
    momentum_normalized = (momentum + 100.0) / 2.0
    return (
        weights["maturity"] * maturity
        + weights["breadth"] * breadth
        + weights["momentum"] * momentum_normalized
    )


@pytest.fixture
def sample_scores() -> pd.DataFrame:
    """A small, easy-to-hand-compute scores frame."""

    return pd.DataFrame(
        [
            {"Ticker": "AAA", "Bank": "Alpha Bank", "Maturity": 80.0, "Breadth": 60.0, "Momentum": 20.0},
            {"Ticker": "BBB", "Bank": "Bravo Bank", "Maturity": 50.0, "Breadth": 40.0, "Momentum": -20.0},
            {"Ticker": "CCC", "Bank": "Charlie Bank", "Maturity": 30.0, "Breadth": 90.0, "Momentum": 0.0},
        ]
    )


@pytest.fixture
def abc_corner_frame() -> pd.DataFrame:
    """Three banks that each max out a single score dimension."""

    return pd.DataFrame(
        [
            {"Ticker": "AAA", "Bank": "MaturityMax", "Maturity": 100.0, "Breadth": 0.0, "Momentum": 0.0},
            {"Ticker": "BBB", "Bank": "MomentumMax", "Maturity": 0.0, "Breadth": 0.0, "Momentum": 100.0},
            {"Ticker": "CCC", "Bank": "BreadthMax", "Maturity": 0.0, "Breadth": 100.0, "Momentum": 0.0},
        ]
    )


def test_default_weights_reproduce_canonical_composite(sample_scores: pd.DataFrame) -> None:
    result = recompute_scores(sample_scores, DEFAULT_WEIGHTS)

    for _, row in sample_scores.iterrows():
        expected = _canonical_composite(
            row["Maturity"], row["Breadth"], row["Momentum"], DEFAULT_WEIGHTS
        )
        got = result.loc[result["Ticker"] == row["Ticker"], "Composite"].iloc[0]
        assert abs(got - expected) < 0.01


def test_pure_maturity_weight(sample_scores: pd.DataFrame) -> None:
    result = recompute_scores(
        sample_scores, {"maturity": 1, "breadth": 0, "momentum": 0}
    )
    for _, row in result.iterrows():
        assert abs(row["Composite"] - row["Maturity"]) < 0.01


def test_weights_are_normalized(sample_scores: pd.DataFrame) -> None:
    # 100:70:30 is the same proportion as 50:35:15.
    scaled = recompute_scores(
        sample_scores, {"maturity": 100, "breadth": 70, "momentum": 30}
    )
    baseline = recompute_scores(sample_scores, DEFAULT_WEIGHTS)

    for ticker in sample_scores["Ticker"]:
        a = scaled.loc[scaled["Ticker"] == ticker, "Composite"].iloc[0]
        b = baseline.loc[baseline["Ticker"] == ticker, "Composite"].iloc[0]
        assert abs(a - b) < 0.01


def test_all_zero_weights_falls_back_to_defaults(sample_scores: pd.DataFrame) -> None:
    zeroed = recompute_scores(
        sample_scores, {"maturity": 0, "breadth": 0, "momentum": 0}
    )
    baseline = recompute_scores(sample_scores, DEFAULT_WEIGHTS)

    for ticker in sample_scores["Ticker"]:
        a = zeroed.loc[zeroed["Ticker"] == ticker, "Composite"].iloc[0]
        b = baseline.loc[baseline["Ticker"] == ticker, "Composite"].iloc[0]
        assert abs(a - b) < 0.01


def test_rank_recomputes_on_weight_change(abc_corner_frame: pd.DataFrame) -> None:
    default_result = recompute_scores(abc_corner_frame, DEFAULT_WEIGHTS)
    top_default = default_result.loc[default_result["Rank"] == 1, "Ticker"].iloc[0]
    assert top_default == "AAA"  # maturity-max wins under defaults

    momentum_result = recompute_scores(
        abc_corner_frame, {"maturity": 0, "breadth": 0, "momentum": 1}
    )
    top_momentum = momentum_result.loc[momentum_result["Rank"] == 1, "Ticker"].iloc[0]
    assert top_momentum == "BBB"

    breadth_result = recompute_scores(
        abc_corner_frame, {"maturity": 0, "breadth": 1, "momentum": 0}
    )
    top_breadth = breadth_result.loc[breadth_result["Rank"] == 1, "Ticker"].iloc[0]
    assert top_breadth == "CCC"


def test_empty_frame_returns_empty() -> None:
    empty = pd.DataFrame()
    result = recompute_scores(empty, DEFAULT_WEIGHTS)
    assert isinstance(result, pd.DataFrame)
    assert result.empty


def test_missing_columns_returns_input_unchanged() -> None:
    frame = pd.DataFrame(
        [
            {"Ticker": "AAA", "Bank": "Alpha", "Maturity": 80.0},
            {"Ticker": "BBB", "Bank": "Bravo", "Maturity": 50.0},
        ]
    )
    result = recompute_scores(frame, DEFAULT_WEIGHTS)
    # Should not raise and should still be a DataFrame with the input columns.
    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == list(frame.columns)
    pd.testing.assert_frame_equal(result.reset_index(drop=True), frame.reset_index(drop=True))


def test_does_not_mutate_input(sample_scores: pd.DataFrame) -> None:
    snapshot = sample_scores.copy(deep=True)
    _ = recompute_scores(
        sample_scores, {"maturity": 2, "breadth": 1, "momentum": 1}
    )
    pd.testing.assert_frame_equal(sample_scores, snapshot)


def test_normalize_sums_to_one() -> None:
    normalized = normalize_weights(
        {"maturity": 25, "breadth": 25, "momentum": 50}
    )
    assert abs(sum(normalized.values()) - 1.0) < 1e-9
    assert abs(normalized["maturity"] - 0.25) < 1e-9
    assert abs(normalized["breadth"] - 0.25) < 1e-9
    assert abs(normalized["momentum"] - 0.50) < 1e-9


def test_normalize_all_zero_returns_defaults() -> None:
    result = normalize_weights({"maturity": 0, "breadth": 0, "momentum": 0})
    assert result == DEFAULT_WEIGHTS


def test_format_weights_caption_default() -> None:
    caption = format_weights_caption(DEFAULT_WEIGHTS)
    assert "Maturity 50%" in caption
    assert "Breadth 35%" in caption
    assert "Momentum 15%" in caption
