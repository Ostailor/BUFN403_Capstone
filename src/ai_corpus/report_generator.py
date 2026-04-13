"""Generate a static Markdown report with embedded matplotlib charts
for the AI Intent Classification system."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .config import CorpusPaths, INTENT_LEVELS, APP_CATEGORIES  # noqa: E402

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read_csv_safe(path: Path) -> Optional[pd.DataFrame]:
    """Return a DataFrame or *None* if the file is missing / empty."""
    if not path.exists():
        logger.warning("CSV not found: %s", path)
        return None
    try:
        df = pd.read_csv(path)
        if df.empty:
            logger.warning("CSV is empty: %s", path)
            return None
        return df
    except Exception as exc:
        logger.warning("Failed to read %s: %s", path, exc)
        return None


# ---------------------------------------------------------------------------
# Chart generators
# ---------------------------------------------------------------------------


def _chart_composite_rankings(df: pd.DataFrame, plots_dir: Path) -> str:
    """Horizontal bar chart of top 15 banks by composite score."""
    fname = "intent_composite_rankings.png"
    top = df.nsmallest(15, "Rank").sort_values("Composite", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(top["Ticker"], top["Composite"], color="#4C72B0")
    ax.set_xlabel("Composite Score")
    ax.set_title("Top 15 Banks by AI Intent Composite Score")
    fig.tight_layout()
    fig.savefig(plots_dir / fname, dpi=150)
    plt.close(fig)
    return fname


def _chart_maturity_vs_breadth(df: pd.DataFrame, plots_dir: Path) -> str:
    """Scatter plot: x=Maturity, y=Breadth, labeled with tickers."""
    fname = "intent_maturity_vs_breadth.png"

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(df["Maturity"], df["Breadth"], alpha=0.7, s=60, color="#DD8452")
    for _, row in df.iterrows():
        ax.annotate(
            row["Ticker"],
            (row["Maturity"], row["Breadth"]),
            fontsize=7,
            ha="left",
            va="bottom",
        )
    ax.set_xlabel("Maturity Score")
    ax.set_ylabel("Breadth Score")
    ax.set_title("AI Intent Maturity vs. Breadth")
    fig.tight_layout()
    fig.savefig(plots_dir / fname, dpi=150)
    plt.close(fig)
    return fname


def _chart_momentum(df: pd.DataFrame, plots_dir: Path) -> str:
    """Horizontal bar chart of momentum scores, green/red coloring."""
    fname = "intent_momentum_bar.png"
    sorted_df = df.sort_values("Momentum", ascending=True)

    colors = ["#55A868" if v >= 0 else "#C44E52" for v in sorted_df["Momentum"]]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(sorted_df["Ticker"], sorted_df["Momentum"], color=colors)
    ax.axvline(0, color="grey", linewidth=0.8)
    ax.set_xlabel("Momentum Score")
    ax.set_title("AI Intent Momentum by Bank")
    fig.tight_layout()
    fig.savefig(plots_dir / fname, dpi=150)
    plt.close(fig)
    return fname


def _chart_category_heatmap(
    df: pd.DataFrame, plots_dir: Path
) -> str:
    """Heatmap of banks x categories (mention count) using imshow."""
    fname = "intent_category_heatmap.png"

    pivot = df.pivot_table(
        index="Ticker", columns="Category", values="Mention_Count", fill_value=0
    )

    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd")

    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(pivot.index, fontsize=8)
    ax.set_title("AI Application Category Mentions by Bank")
    fig.colorbar(im, ax=ax, label="Mention Count")
    fig.tight_layout()
    fig.savefig(plots_dir / fname, dpi=150)
    plt.close(fig)
    return fname


# ---------------------------------------------------------------------------
# Markdown builders
# ---------------------------------------------------------------------------


def _section_executive_summary(composite_df: pd.DataFrame) -> str:
    top5 = composite_df.nsmallest(5, "Rank")
    avg_maturity = composite_df["Maturity"].mean()
    lines = [
        "## 1. Executive Summary\n",
        f"This report analyzes AI intent classification across **{len(composite_df)} banks**.\n",
        f"**Average maturity score:** {avg_maturity:.2f} (scale 1–4)\n",
        "### Top 5 Banks by Composite Score\n",
        "| Rank | Ticker | Bank | Composite | Maturity | Breadth | Momentum |",
        "|------|--------|------|-----------|----------|---------|----------|",
    ]
    for _, r in top5.iterrows():
        lines.append(
            f"| {int(r['Rank'])} | {r['Ticker']} | {r['Bank']} | "
            f"{r['Composite']:.3f} | {r['Maturity']:.2f} | "
            f"{r['Breadth']:.2f} | {r['Momentum']:.2f} |"
        )
    lines.append("")
    return "\n".join(lines)


def _section_methodology() -> str:
    intent_desc = ", ".join(f"**{v}** ({k})" for k, v in INTENT_LEVELS.items())
    cat_list = ", ".join(f"*{c}*" for c in APP_CATEGORIES)
    return "\n".join([
        "## 2. Methodology\n",
        "### Intent Levels",
        f"Each AI mention is classified into one of four intent levels: {intent_desc}.\n",
        "### Application Categories",
        f"Mentions are mapped to six application categories: {cat_list}.\n",
        "### Scoring Formulas",
        "- **Maturity** — Weighted average of intent levels across all mentions.",
        "- **Breadth** — Number of distinct application categories with at least one mention.",
        "- **Momentum** — Change in maturity score between the first and last observed quarters.",
        "- **Composite** — Normalized weighted combination of Maturity, Breadth, and Momentum.",
        "",
    ])


def _section_full_rankings(composite_df: pd.DataFrame) -> str:
    sorted_df = composite_df.sort_values("Rank")
    lines = [
        "## 3. Full Rankings\n",
        "| Rank | Ticker | Bank | Composite | Maturity | Breadth | Momentum |",
        "|------|--------|------|-----------|----------|---------|----------|",
    ]
    for _, r in sorted_df.iterrows():
        lines.append(
            f"| {int(r['Rank'])} | {r['Ticker']} | {r['Bank']} | "
            f"{r['Composite']:.3f} | {r['Maturity']:.2f} | "
            f"{r['Breadth']:.2f} | {r['Momentum']:.2f} |"
        )
    lines.append("")
    return "\n".join(lines)


def _section_visualizations(chart_files: list[str]) -> str:
    lines = ["## 4. Key Visualizations\n"]
    titles = {
        "intent_composite_rankings.png": "Composite Score Rankings",
        "intent_maturity_vs_breadth.png": "Maturity vs. Breadth",
        "intent_momentum_bar.png": "Momentum Scores",
        "intent_category_heatmap.png": "Application Category Heatmap",
    }
    for fname in chart_files:
        title = titles.get(fname, fname)
        lines.append(f"### {title}\n")
        lines.append(f"![{title}](plots/{fname})\n")
    return "\n".join(lines)


def _section_bank_highlights(composite_df: pd.DataFrame) -> str:
    top3 = composite_df.nsmallest(3, "Rank")
    lines = ["## 5. Bank Highlights\n"]
    for _, r in top3.iterrows():
        lines.append(f"### {r['Bank']} ({r['Ticker']})\n")
        lines.append(
            f"- **Rank:** {int(r['Rank'])}  |  "
            f"**Composite:** {r['Composite']:.3f}  |  "
            f"**Maturity:** {r['Maturity']:.2f}  |  "
            f"**Breadth:** {r['Breadth']:.2f}  |  "
            f"**Momentum:** {r['Momentum']:.2f}"
        )
        if r["Momentum"] > 0:
            lines.append(f"- Positive momentum indicates increasing AI commitment over recent quarters.")
        elif r["Momentum"] < 0:
            lines.append(f"- Negative momentum suggests a plateau or pull-back in AI initiatives.")
        else:
            lines.append(f"- Momentum is neutral — steady-state AI activity.")
        lines.append("")
    return "\n".join(lines)


def _section_trends(
    composite_df: pd.DataFrame,
    quarterly_df: Optional[pd.DataFrame],
) -> str:
    lines = ["## 6. Trends & Momentum\n"]

    # Highest / lowest momentum
    highest = composite_df.loc[composite_df["Momentum"].idxmax()]
    lowest = composite_df.loc[composite_df["Momentum"].idxmin()]
    lines.append(
        f"- **Highest momentum:** {highest['Ticker']} ({highest['Momentum']:.2f})"
    )
    lines.append(
        f"- **Lowest momentum:** {lowest['Ticker']} ({lowest['Momentum']:.2f})\n"
    )

    if quarterly_df is not None:
        # Quarter-over-quarter average maturity
        qoq = (
            quarterly_df.groupby(["Year", "Quarter"])["Maturity_Score"]
            .mean()
            .reset_index()
            .sort_values(["Year", "Quarter"])
        )
        lines.append("### Average Maturity by Quarter\n")
        lines.append("| Year | Quarter | Avg Maturity |")
        lines.append("|------|---------|-------------|")
        for _, r in qoq.iterrows():
            lines.append(
                f"| {int(r['Year'])} | Q{int(r['Quarter'])} | {r['Maturity_Score']:.2f} |"
            )
        lines.append("")

        # Scaling percentage trend
        if "Scaling_Pct" in quarterly_df.columns:
            scale_qoq = (
                quarterly_df.groupby(["Year", "Quarter"])["Scaling_Pct"]
                .mean()
                .reset_index()
                .sort_values(["Year", "Quarter"])
            )
            if len(scale_qoq) >= 2:
                first_val = scale_qoq.iloc[0]["Scaling_Pct"]
                last_val = scale_qoq.iloc[-1]["Scaling_Pct"]
                delta = last_val - first_val
                direction = "increased" if delta > 0 else "decreased"
                lines.append(
                    f"Average *Scaling* percentage {direction} from "
                    f"{first_val:.1f}% to {last_val:.1f}% over the observation window.\n"
                )
    else:
        lines.append("*Quarterly progression data not available.*\n")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def generate_report(paths: CorpusPaths) -> Path:
    """Generate the full AI Intent Classification Markdown report.

    Returns the path to the written report file.
    """
    plots_dir = paths.plots_dir
    plots_dir.mkdir(parents=True, exist_ok=True)

    output_dir = paths.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Read input CSVs ───────────────────────────────────────────────
    composite_df = _read_csv_safe(output_dir / "bank_composite_scores.csv")
    quarterly_df = _read_csv_safe(output_dir / "quarterly_progression.csv")
    category_df = _read_csv_safe(output_dir / "app_category_matrix.csv")

    if composite_df is None:
        # Without the main rankings we can only write a stub report.
        report_path = output_dir / "ai_intent_classification_report.md"
        report_path.write_text(
            "# AI Intent Classification Report\n\n"
            "*Report could not be generated — `bank_composite_scores.csv` is missing or empty.*\n"
        )
        logger.warning("Composite scores CSV missing; wrote stub report.")
        return report_path

    # ── Generate charts ───────────────────────────────────────────────
    chart_files: list[str] = []

    try:
        chart_files.append(_chart_composite_rankings(composite_df, plots_dir))
    except Exception as exc:
        logger.warning("Failed to generate composite rankings chart: %s", exc)

    try:
        chart_files.append(_chart_maturity_vs_breadth(composite_df, plots_dir))
    except Exception as exc:
        logger.warning("Failed to generate maturity-vs-breadth chart: %s", exc)

    try:
        chart_files.append(_chart_momentum(composite_df, plots_dir))
    except Exception as exc:
        logger.warning("Failed to generate momentum chart: %s", exc)

    if category_df is not None:
        try:
            chart_files.append(_chart_category_heatmap(category_df, plots_dir))
        except Exception as exc:
            logger.warning("Failed to generate category heatmap: %s", exc)

    # ── Assemble report ───────────────────────────────────────────────
    sections: list[str] = [
        "# AI Intent Classification Report\n",
        _section_executive_summary(composite_df),
        _section_methodology(),
        _section_full_rankings(composite_df),
    ]

    if chart_files:
        sections.append(_section_visualizations(chart_files))

    sections.append(_section_bank_highlights(composite_df))
    sections.append(_section_trends(composite_df, quarterly_df))

    report_text = "\n".join(sections)

    report_path = output_dir / "ai_intent_classification_report.md"
    report_path.write_text(report_text)
    logger.info("Report written to %s", report_path)
    return report_path
