from __future__ import annotations

import csv
import json
import logging
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .config import APP_CATEGORIES, INTENT_LEVELS, CorpusPaths

log = logging.getLogger(__name__)

NUM_APP_CATEGORIES = len(APP_CATEGORIES)


def compute_maturity_score(classifications: list[dict[str, Any]]) -> float:
    if not classifications:
        return 0.0
    total = sum(c["intent_level"] for c in classifications)
    count = len(classifications)
    return total / count * 25


def compute_breadth_score(classifications: list[dict[str, Any]]) -> float:
    if not classifications:
        return 0.0
    category_counts: Counter[str] = Counter()
    for c in classifications:
        for cat in c.get("app_categories", []):
            category_counts[cat] += 1
    total = sum(category_counts.values())
    if total == 0:
        return 0.0
    max_entropy = math.log2(NUM_APP_CATEGORIES)  # log2(6) ≈ 2.585
    entropy = -sum((n / total) * math.log2(n / total) for n in category_counts.values())
    return round(entropy / max_entropy * 100, 2)


def compute_momentum_score(quarterly_maturity: dict[str, float]) -> float:
    if len(quarterly_maturity) < 2:
        return 0.0
    sorted_quarters = sorted(quarterly_maturity.keys())
    values = [quarterly_maturity[q] for q in sorted_quarters]
    n = len(values)
    xs = list(range(n))
    x_bar = sum(xs) / n
    y_bar = sum(values) / n
    numerator = sum((x - x_bar) * (y - y_bar) for x, y in zip(xs, values))
    denominator = sum((x - x_bar) ** 2 for x in xs)
    slope = numerator / denominator if denominator else 0.0
    annualized = slope * 4
    return max(-100.0, min(100.0, round(annualized, 2)))


def compute_composite(maturity: float, breadth: float, momentum: float) -> float:
    momentum_normalized = (momentum + 100) / 2
    return 0.50 * maturity + 0.35 * breadth + 0.15 * momentum_normalized


def run_scoring(paths: CorpusPaths) -> dict[str, Path]:
    if not paths.classifications_jsonl.exists():
        raise FileNotFoundError(f"Classifications not found: {paths.classifications_jsonl}")

    classifications: list[dict[str, Any]] = []
    for line in paths.classifications_jsonl.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            classifications.append(json.loads(line))

    log.info("Loaded %d classifications", len(classifications))

    by_bank: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for c in classifications:
        by_bank[c["ticker"]].append(c)

    # Compute per-bank quarterly maturity for momentum
    bank_quarterly: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for c in classifications:
        y = c.get("period_year")
        q = c.get("period_quarter")
        if y and q:
            key = f"{y}_Q{q}"
            bank_quarterly[c["ticker"]][key].append(c["intent_level"])

    composite_rows: list[dict[str, Any]] = []
    quarterly_rows: list[dict[str, Any]] = []
    category_rows: list[dict[str, Any]] = []

    for ticker, bank_classifications in sorted(by_bank.items()):
        bank_name = bank_classifications[0].get("bank_name", ticker)
        maturity = compute_maturity_score(bank_classifications)
        breadth = compute_breadth_score(bank_classifications)

        # Quarterly maturity scores
        quarterly_maturity: dict[str, float] = {}
        for qkey, levels in sorted(bank_quarterly[ticker].items()):
            q_mat = sum(levels) / len(levels) * 25
            quarterly_maturity[qkey] = q_mat

            # Intent distribution for this quarter
            level_counts = Counter(levels)
            total = len(levels)
            quarterly_rows.append({
                "Ticker": ticker,
                "Year": int(qkey.split("_Q")[0]),
                "Quarter": int(qkey.split("_Q")[1]),
                "Maturity_Score": round(q_mat, 2),
                "Exploring_Pct": round(level_counts.get(1, 0) / total * 100, 1),
                "Committing_Pct": round(level_counts.get(2, 0) / total * 100, 1),
                "Deploying_Pct": round(level_counts.get(3, 0) / total * 100, 1),
                "Scaling_Pct": round(level_counts.get(4, 0) / total * 100, 1),
            })

        momentum = compute_momentum_score(quarterly_maturity)
        composite = compute_composite(maturity, breadth, momentum)

        composite_rows.append({
            "Ticker": ticker,
            "Bank": bank_name,
            "Maturity": round(maturity, 2),
            "Breadth": round(breadth, 2),
            "Momentum": round(momentum, 2),
            "Composite": round(composite, 2),
        })

        # App category matrix
        cat_counts: Counter[str] = Counter()
        cat_levels: dict[str, list[int]] = defaultdict(list)
        for c in bank_classifications:
            for cat in c.get("app_categories", []):
                cat_counts[cat] += 1
                cat_levels[cat].append(c["intent_level"])
        for cat in sorted(set(cat_counts.keys()) | set(APP_CATEGORIES.keys())):
            category_rows.append({
                "Ticker": ticker,
                "Category": cat,
                "Mention_Count": cat_counts.get(cat, 0),
                "Avg_Intent_Level": round(
                    sum(cat_levels[cat]) / len(cat_levels[cat]), 2
                ) if cat_levels.get(cat) else 0.0,
            })

    # Rank by composite
    composite_rows.sort(key=lambda r: r["Composite"], reverse=True)
    for i, row in enumerate(composite_rows, 1):
        row["Rank"] = i

    # Write outputs
    paths.output_dir.mkdir(parents=True, exist_ok=True)

    scores_csv = paths.output_dir / "bank_composite_scores.csv"
    _write_csv(scores_csv, composite_rows)
    log.info("Wrote %s (%d banks)", scores_csv, len(composite_rows))

    progression_csv = paths.output_dir / "quarterly_progression.csv"
    _write_csv(progression_csv, quarterly_rows)
    log.info("Wrote %s (%d rows)", progression_csv, len(quarterly_rows))

    matrix_csv = paths.output_dir / "app_category_matrix.csv"
    _write_csv(matrix_csv, category_rows)
    log.info("Wrote %s (%d rows)", matrix_csv, len(category_rows))

    return {
        "bank_composite_scores": scores_csv,
        "quarterly_progression": progression_csv,
        "app_category_matrix": matrix_csv,
        "banks_scored": len(composite_rows),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
