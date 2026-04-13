from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ai_corpus.composite_scorer import (
    compute_breadth_score,
    compute_composite,
    compute_maturity_score,
    compute_momentum_score,
)


def test_maturity_score_basic() -> None:
    classifications = [
        {"intent_level": 3},
        {"intent_level": 3},
        {"intent_level": 2},
        {"intent_level": 1},
    ]
    result = compute_maturity_score(classifications)
    # sum = 2*3 + 1*2 + 1*1 = 9, count = 4, score = 9/4 * 25 = 56.25
    assert result == 56.25


def test_maturity_score_all_scaling() -> None:
    classifications = [
        {"intent_level": 4},
        {"intent_level": 4},
        {"intent_level": 4},
    ]
    result = compute_maturity_score(classifications)
    # avg = 4, score = 4/1 * 25 = 100.0  (i.e. avg/max_level * 100)
    assert result == 100.0


def test_maturity_score_all_exploring() -> None:
    classifications = [
        {"intent_level": 1},
        {"intent_level": 1},
        {"intent_level": 1},
    ]
    result = compute_maturity_score(classifications)
    # avg = 1, score = 1/4 * 100 = 25.0
    assert result == 25.0


def test_breadth_score_perfectly_even() -> None:
    # 6 categories, equal mentions → max entropy → 100.0
    categories = ["A", "B", "C", "D", "E", "F"]
    classifications = [{"app_categories": [cat]} for cat in categories for _ in range(5)]
    result = compute_breadth_score(classifications)
    assert result == 100.0


def test_breadth_score_single_category() -> None:
    # All mentions in one category → zero entropy → 0.0
    classifications = [{"app_categories": ["GenAI / LLMs"]} for _ in range(20)]
    result = compute_breadth_score(classifications)
    assert result == 0.0


def test_breadth_score_partial_even() -> None:
    # 3 categories, equal mentions → log2(3)/log2(6) * 100
    classifications = [
        {"app_categories": ["GenAI / LLMs"]},
        {"app_categories": ["GenAI / LLMs"]},
        {"app_categories": ["Predictive ML"]},
        {"app_categories": ["Predictive ML"]},
        {"app_categories": ["Fraud / Risk Models"]},
        {"app_categories": ["Fraud / Risk Models"]},
    ]
    result = compute_breadth_score(classifications)
    assert result == 61.31


def test_breadth_score_skewed() -> None:
    # Heavy GenAI, light others (like NTRS: 44,6,4,3,2,2)
    classifications = (
        [{"app_categories": ["GenAI / LLMs"]}] * 44
        + [{"app_categories": ["RPA / Automation"]}] * 6
        + [{"app_categories": ["Fraud / Risk Models"]}] * 4
        + [{"app_categories": ["Computer Vision"]}] * 3
        + [{"app_categories": ["NLP / Text"]}] * 2
        + [{"app_categories": ["Predictive ML"]}] * 2
    )
    result = compute_breadth_score(classifications)
    assert result == 56.63


def test_breadth_score_no_mentions() -> None:
    result = compute_breadth_score([])
    assert result == 0.0


def test_momentum_positive_trend() -> None:
    quarterly_maturity = {
        "2024_Q1": 40, "2024_Q2": 45, "2024_Q3": 55, "2024_Q4": 60,
    }
    result = compute_momentum_score(quarterly_maturity)
    # slope = 7.0 per quarter, annualized = 28.0
    assert result == 28.0


def test_momentum_negative_trend() -> None:
    quarterly_maturity = {
        "2024_Q1": 60, "2024_Q2": 55, "2024_Q3": 40, "2024_Q4": 35,
    }
    result = compute_momentum_score(quarterly_maturity)
    # slope = -9.0 per quarter, annualized = -36.0
    assert result == -36.0


def test_momentum_flat() -> None:
    quarterly_maturity = {
        "2024_Q1": 50, "2024_Q2": 50, "2024_Q3": 50, "2024_Q4": 50,
    }
    result = compute_momentum_score(quarterly_maturity)
    assert result == 0.0


def test_momentum_insufficient_data() -> None:
    quarterly_maturity = {"2024_Q1": 50}
    result = compute_momentum_score(quarterly_maturity)
    assert result == 0.0


def test_momentum_two_quarters_clamped() -> None:
    quarterly_maturity = {"2024_Q1": 30, "2024_Q2": 60}
    result = compute_momentum_score(quarterly_maturity)
    # slope = 30 per quarter, annualized = 120, clamped to 100.0
    assert result == 100.0


def test_composite_formula() -> None:
    # momentum_normalized = (20 + 100) / 2 = 60
    # composite = 0.50*80 + 0.35*60 + 0.15*60 = 40 + 21 + 9 = 70.0
    result = compute_composite(maturity=80, breadth=60, momentum=20)
    assert result == 70.0


def test_composite_negative_momentum() -> None:
    # momentum_normalized = (-50 + 100) / 2 = 25
    # composite = 0.50*50 + 0.35*50 + 0.15*25 = 25 + 17.5 + 3.75 = 46.25
    result = compute_composite(maturity=50, breadth=50, momentum=-50)
    assert result == 46.25


def test_composite_zero_momentum() -> None:
    # momentum_normalized = (0 + 100) / 2 = 50
    # composite = 0.50*60 + 0.35*40 + 0.15*50 = 30 + 14 + 7.5 = 51.5
    result = compute_composite(maturity=60, breadth=40, momentum=0)
    assert result == 51.5
