#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def ensure_columns(frame: pd.DataFrame, required: list[str]) -> None:
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise SystemExit(f"Missing required columns in CSV: {', '.join(missing)}")


def plot_score_distributions(frame: pd.DataFrame, out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(10, 6))
    bins = [x / 2 for x in range(2, 22)]  # 1.0 to 10.5 in 0.5 steps
    ax.hist(frame["AI_Score"], bins=bins, alpha=0.55, label="AI_Score (Hybrid)", color="#1f77b4")
    ax.hist(frame["Rule_AI_Score"], bins=bins, alpha=0.45, label="Rule_AI_Score", color="#2ca02c")
    ax.hist(frame["LLM_AI_Score"], bins=bins, alpha=0.45, label="LLM_AI_Score", color="#ff7f0e")
    ax.set_title("Score Distributions")
    ax.set_xlabel("Score")
    ax.set_ylabel("Bank count")
    ax.set_xlim(1.0, 10.0)
    ax.grid(axis="y", alpha=0.2)
    ax.legend()
    output = out_dir / "01_score_distributions.png"
    fig.tight_layout()
    fig.savefig(output, dpi=170)
    plt.close(fig)
    return output


def plot_rule_vs_llm(frame: pd.DataFrame, out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(frame["Rule_AI_Score"], frame["LLM_AI_Score"], color="#9467bd", alpha=0.8, s=42)
    ax.plot([1, 10], [1, 10], linestyle="--", color="#555555", linewidth=1.2, label="y = x")
    ax.set_title("Rule vs LLM Score by Bank")
    ax.set_xlabel("Rule_AI_Score")
    ax.set_ylabel("LLM_AI_Score")
    ax.set_xlim(1.0, 10.0)
    ax.set_ylim(1.0, 10.0)
    ax.grid(alpha=0.2)
    ax.legend(loc="upper left")
    output = out_dir / "02_rule_vs_llm_scatter.png"
    fig.tight_layout()
    fig.savefig(output, dpi=170)
    plt.close(fig)
    return output


def plot_top_banks(frame: pd.DataFrame, out_dir: Path, top_n: int) -> Path:
    top = frame.sort_values("AI_Score", ascending=False).head(top_n).copy()
    top = top.sort_values("AI_Score", ascending=True)
    labels = [f"{row.Ticker} ({row.Bank[:18]})" for _, row in top.iterrows()]

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.barh(labels, top["AI_Score"], color="#1f77b4", alpha=0.85, label="AI_Score")
    ax.scatter(top["Rule_AI_Score"], labels, color="#2ca02c", marker="o", s=40, label="Rule_AI_Score")
    ax.scatter(top["LLM_AI_Score"], labels, color="#ff7f0e", marker="x", s=48, label="LLM_AI_Score")
    ax.set_title(f"Top {top_n} Banks by AI_Score")
    ax.set_xlabel("Score")
    ax.set_xlim(1.0, 10.0)
    ax.grid(axis="x", alpha=0.2)
    ax.legend(loc="lower right")
    output = out_dir / "03_top_banks_ai_score.png"
    fig.tight_layout()
    fig.savefig(output, dpi=170)
    plt.close(fig)
    return output


def plot_rule_llm_gap(frame: pd.DataFrame, out_dir: Path, top_n: int) -> Path:
    gap = frame.copy()
    gap["Rule_minus_LLM"] = gap["Rule_AI_Score"] - gap["LLM_AI_Score"]
    gap = gap.sort_values("Rule_minus_LLM", ascending=False).head(top_n)
    gap = gap.sort_values("Rule_minus_LLM", ascending=True)

    fig, ax = plt.subplots(figsize=(11, 7))
    colors = ["#d62728" if value > 0 else "#17becf" for value in gap["Rule_minus_LLM"]]
    ax.barh(gap["Ticker"], gap["Rule_minus_LLM"], color=colors, alpha=0.85)
    ax.axvline(0.0, color="#555555", linewidth=1.0)
    ax.set_title(f"Largest Rule-LLM Gaps (Top {top_n})")
    ax.set_xlabel("Rule_AI_Score - LLM_AI_Score")
    ax.set_ylabel("Ticker")
    ax.grid(axis="x", alpha=0.2)
    output = out_dir / "04_rule_minus_llm_gap.png"
    fig.tight_layout()
    fig.savefig(output, dpi=170)
    plt.close(fig)
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create diagnostic plots for AI bank classification results.")
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path("/Users/omtailor/BUFN403_Capstone/AI_Bank_Classification.csv"),
        help="Path to AI_Bank_Classification.csv",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/Users/omtailor/BUFN403_Capstone/plots"),
        help="Directory where plot images will be saved",
    )
    parser.add_argument("--top-n", type=int, default=15, help="Number of banks to include in top-bank/gap charts")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if not args.input_csv.exists():
        raise SystemExit(f"Input CSV not found: {args.input_csv}")

    frame = pd.read_csv(args.input_csv)
    ensure_columns(frame, ["Bank", "Ticker", "AI_Score", "Rule_AI_Score", "LLM_AI_Score"])

    args.out_dir.mkdir(parents=True, exist_ok=True)
    outputs = [
        plot_score_distributions(frame, args.out_dir),
        plot_rule_vs_llm(frame, args.out_dir),
        plot_top_banks(frame, args.out_dir, top_n=max(5, args.top_n)),
        plot_rule_llm_gap(frame, args.out_dir, top_n=max(5, args.top_n)),
    ]

    print("Generated plot files:")
    for path in outputs:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
