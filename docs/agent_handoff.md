# Agent Handoff — Current Project State

**Last updated:** April 13, 2026

## What This Project Is

BUFN403 Capstone analyzing AI adoption across 50 U.S. banks. Uses earnings transcripts, SEC filings, and FDIC call reports. The team subtopic is **Classification and Intent**.

## What Has Been Built

### Core Pipeline (pre-existing)
- RAG pipeline: chunks.jsonl → Chroma index → Qwen Q&A (`src/ai_corpus/pipeline.py`)
- Rule + LLM hybrid scorer with 1-10 scale (`scripts/classify_bank_ai.py`)
- Theme tagging across 9 categories (`src/ai_corpus/config.py`)

### New: Intent Classification System (completed April 13, 2026)

**Files created:**
| File | Purpose |
|------|---------|
| `src/ai_corpus/intent_classifier.py` | Classifies chunks by intent level (1-4) and app category (6 types) |
| `src/ai_corpus/composite_scorer.py` | Computes Maturity, Breadth, Momentum scores; ranks banks |
| `src/ai_corpus/report_generator.py` | Generates Markdown report with matplotlib charts |
| `dashboard/app.py` | Streamlit entry point |
| `dashboard/data_loader.py` | Shared data loading with cache + fallback to mock data |
| `dashboard/pages/1_Leaderboard.py` | Ranked bank table with filters |
| `dashboard/pages/2_Bank_Deep_Dive.py` | Per-bank charts and evidence |
| `dashboard/pages/3_Compare_Banks.py` | Side-by-side bank comparison |
| `dashboard/pages/4_Market_Overview.py` | Market-wide heatmaps and quadrant |
| `tests/test_intent_classifier.py` | 8 tests for classifier |
| `tests/test_composite_scorer.py` | 11 tests for scorer |
| `presentation.html` | 11-slide HTML presentation for class demo |
| `docs/rule_based_classifier.md` | Documents the rule-based classification approach |

**Files modified:**
| File | Change |
|------|--------|
| `src/ai_corpus/config.py` | Added INTENT_LEVELS, INTENT_SIGNAL_WORDS, APP_CATEGORIES, classifications_jsonl path |
| `src/ai_corpus/__init__.py` | Exported run_classification, run_scoring, generate_report |
| `scripts/ai_corpus_pipeline.py` | Added classify, score, report, dashboard subcommands |
| `requirements.txt` | Added streamlit, plotly |

## Current Data State

All artifacts have been generated and are in `artifacts/ai_corpus/`:

- `chunks.jsonl` — 256,385 chunks (Git LFS, ~410MB)
- `classifications.jsonl` — 2,757 AI chunk classifications (rule-based, confidence=0.5)
- `bank_composite_scores.csv` — 50 banks scored and ranked
- `quarterly_progression.csv` — per-bank per-quarter maturity time series
- `app_category_matrix.csv` — per-bank per-category mention counts
- `ai_intent_classification_report.md` — static Markdown report
- `plots/intent_*.png` — 4 classification charts

## What Could Be Done Next

1. **LLM classification** — The Qwen API calls in `classify_chunk()` were failing. Debug the HF API integration or switch to a different model. Rule-based fallback is working but less nuanced.

**Note:** Breadth score uses **Shannon entropy** (normalized 0-100). This penalizes concentration in a single category — a bank heavily skewed to GenAI scores lower than one evenly spread across all 6 categories. See `compute_breadth_score()` in `composite_scorer.py`.

**Note:** Momentum score now uses **OLS linear regression slope** of quarterly maturity scores, annualized and clamped to [-100, +100]. This replaced the earlier half-over-half comparison. See `compute_momentum_score()` in `composite_scorer.py`.

**Note:** Composite weights updated to **0.50 Maturity / 0.35 Breadth / 0.15 Momentum** (previously 0.45/0.30/0.25). Momentum weight reduced because it is the noisiest signal. Full formulas documented in [`docs/scoring_methodology.md`](scoring_methodology.md).
2. ~~**Richer momentum** — Current momentum uses half-over-half comparison. Could do quarter-over-quarter with trend lines.~~ **Done** — momentum now uses linear regression slope.
3. **Dashboard polish** — Add more interactivity, export buttons, custom theming.
4. **PDF report export** — `report_generator.py` outputs Markdown; could add pandoc-based PDF generation.
5. **Benchmark evaluation** — Compare rule-based vs LLM classifications on a labeled test set.
6. **Additional data sources** — Integrate MRA/MRIA supervisory documents (some are in `manual_sources/`).

## How to Run

```bash
cd /Users/adityadabeer/Documents/VSCODE/misc/bufn403/BUFN403_Capstone

# Run tests (all 31 should pass)
/Users/adityadabeer/.pyenv/versions/400/bin/python -m pytest tests/ -v

# Re-run pipeline (if data changes)
python scripts/ai_corpus_pipeline.py classify
python scripts/ai_corpus_pipeline.py score
python scripts/ai_corpus_pipeline.py report

# Launch dashboard
streamlit run dashboard/app.py

# Open presentation
open presentation.html
```

## Python Environment

The working Python environment is `/Users/adityadabeer/.pyenv/versions/400/bin/python` (Python 3.11.5, pytest-8.3.5). The `311` env has dependency issues (numpy/pandas binary mismatch). Git LFS is installed at `/usr/local/bin/git-lfs`.

## Key Architecture Notes

- `CorpusPaths` dataclass (`src/ai_corpus/config.py`) manages all artifact paths
- `QwenAnswerGenerator` (`src/ai_corpus/qwen.py`) handles LLM inference with local/remote fallback
- `has_ai_anchor()` (`src/ai_corpus/pipeline.py`) is the shared AI keyword filter
- CLI uses argparse subcommands in `scripts/ai_corpus_pipeline.py`
- Dashboard falls back to mock data if CSVs don't exist
