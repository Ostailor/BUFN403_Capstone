# Agent Handoff — Current Project State

**Last updated:** April 15, 2026

## Dashboard Plugin System (April 21, 2026)

The dashboard has been refactored into a team-plugin model. `dashboard/app.py`
now discovers teams by scanning `dashboard/teams/<slug>/manifest.toml` at
runtime, and each team owns its own `data_loader.py` and `pages/` directory.
The class is organized as a 3 x 2 matrix (section x sub-team) with six slugs:

- `ai_classification_intent` (reference implementation, Aditya's team)
- `ai_risk`
- `crypto_classification_intent`
- `crypto_risk`
- `private_credit_classification_intent`
- `private_credit_risk`

New contributors (human or agent) should start at `AGENTS.md` in the repo
root, or `docs/dashboard_teams.md` for a narrative overview. Pages are added
with `scripts/new_team_page.py`. The pre-existing AI Classification & Intent
pages (see below) now live under `dashboard/teams/ai_classification_intent/`.

## What This Project Is

BUFN403 Capstone analyzing AI adoption across 50 U.S. banks. Uses earnings transcripts, SEC filings, and FDIC call reports. The team subtopic is **Classification and Intent**.

GitHub repo: `Ostailor/BUFN403_Capstone` (branch: `aditya/ai-intent-classification`)

## What Has Been Built

### Core Pipeline (pre-existing)
- RAG pipeline: chunks.jsonl → Chroma index → Qwen Q&A (`src/ai_corpus/pipeline.py`)
- Rule + LLM hybrid scorer with 1-10 scale (`scripts/classify_bank_ai.py`) — now superseded by intent classifier
- Theme tagging across 9 categories (`src/ai_corpus/config.py`)

### Intent Classification System (completed April 13, 2026)

**Files created:**
| File | Purpose |
|------|---------|
| `src/ai_corpus/intent_classifier.py` | Classifies chunks by intent level (1-4) and app category (6 types); Qwen LLM primary, rule-based reference only |
| `src/ai_corpus/composite_scorer.py` | Computes Maturity, Breadth, Momentum scores; ranks banks |
| `src/ai_corpus/classification_io.py` | Shared classification I/O: JSONL read/write, normalization, confidence parsing, malformed-tail repair |
| `src/ai_corpus/report_generator.py` | Generates Markdown report with matplotlib charts |
| `dashboard/app.py` | Streamlit entry point |
| `dashboard/data_loader.py` | Shared data loading from canonical artifacts (no mock/heuristic fallback) |
| `dashboard/pages/1_Leaderboard.py` | Ranked bank table with filters |
| `dashboard/pages/2_Bank_Deep_Dive.py` | Per-bank charts, evidence, and all 5 metrics (Rank, Composite, Maturity, Breadth, Momentum) |
| `dashboard/pages/3_Compare_Banks.py` | Side-by-side bank comparison |
| `dashboard/pages/4_Market_Overview.py` | Market-wide heatmap (log-scaled, Blues), maturity quadrant (colored by Momentum), intent distribution, top movers |
| `tests/test_intent_classifier.py` | Extensive classifier tests including Qwen output parsing, confidence normalization, truncated JSON recovery, resume logic |
| `tests/test_composite_scorer.py` | Scorer tests including edge cases |
| `tests/test_dashboard_data_loader.py` | Dashboard data loader tests |
| `presentation.html` | 13-slide HTML presentation for class demo |
| `docs/rule_based_classifier.md` | Documents the rule-based classification approach |
| `docs/scoring_methodology.md` | Technical reference for scoring formulas |
| `docs/how_scoring_works.md` | Thorough plain-English walkthrough of the entire scoring pipeline with worked examples |

**Files modified:**
| File | Change |
|------|--------|
| `src/ai_corpus/config.py` | INTENT_LEVELS, INTENT_SIGNAL_WORDS, APP_CATEGORIES, CLASSIFIER_BATCH_SIZE, classification paths |
| `src/ai_corpus/qwen.py` | Qwen model wrapper with fallback chain (7B → 3B → 1.5B → 0.5B), partial-JSON recovery for truncated outputs |
| `scripts/ai_corpus_pipeline.py` | Added classify, score, report, dashboard subcommands; Colab resume flags |
| `requirements.txt` | Added streamlit, plotly |

### Qwen Classification Hardening (completed April 13-15, 2026)

Five commits that made the Qwen classification pipeline reliable on Google Colab:

1. **Fail-fast design** — Classification no longer silently falls back to regex heuristics when Qwen fails. Every classification in the final dataset came from the model.
2. **Confidence normalization** — Handles Qwen returning confidence as text labels ("high", "medium") or percentage strings, not just floats.
3. **Truncated JSON recovery** — When Qwen hits max token limit mid-generation, extracts usable structured fields from incomplete JSON instead of failing the whole run.
4. **Resume support** — Colab runs can stop mid-append and recover: repairs malformed trailing JSONL record, skips completed chunks, exits early when done. Progress/ETA reset per session.
5. **Artifact refresh** — Final data artifacts replaced with completed Qwen run results (2,757 classifications, 50 banks scored).

### Dashboard Visual Fixes (April 15, 2026)

- **Heatmap**: Switched to log-scaled (`log1p`) Blues colorscale; sorted banks by total mentions; hover shows actual counts. Fixed the "wall of yellow" problem caused by outlier mention counts (range 1-261, median 12).
- **Maturity Quadrant**: Removed per-ticker coloring (50 tickers recycling 9 colors). Now colored by Momentum (RdYlGn diverging scale). Uniform dot size instead of oversized Breadth-based bubbles. Added quadrant labels (Leaders, Mature Niche, Broad Early, Lagging).
- **Bank Deep Dive**: Added Composite Score metric alongside Rank, Maturity, Breadth, Momentum.
- **sys.path fix**: All dashboard pages now add repo root to sys.path so `from src.ai_corpus` imports work.

## Current Data State

All artifacts are generated from a completed Qwen classification run and live in `artifacts/ai_corpus/`:

- `chunks.jsonl` — 256,385 chunks (Git LFS, ~410MB)
- `classifications.jsonl` — 2,757 AI chunk classifications (Qwen-backed)
- `bank_composite_scores.csv` — 50 banks scored and ranked (top: Cullen/Frost at 74.38)
- `quarterly_progression.csv` — per-bank per-quarter maturity time series
- `app_category_matrix.csv` — per-bank per-category mention counts
- `ai_intent_classification_report.md` — static Markdown report
- `plots/intent_*.png` — classification charts

Additional artifacts in `artifacts/april_1_ai_team/`:
- Financial & risk integration (AI maturity correlated with assets, profitability, leverage)

## Scoring Summary

- **Maturity** (0-100): `avg(intent_levels) / 4 × 100`
- **Breadth** (0-100): Shannon entropy of 6 app categories, normalized by `log₂(6)`
- **Momentum** (-100 to +100): OLS regression slope of quarterly maturity, annualized (`slope × 4`), clamped
- **Composite** (0-100): `0.50 × Maturity + 0.35 × Breadth + 0.15 × (Momentum + 100) / 2`

Full details: [`docs/how_scoring_works.md`](how_scoring_works.md) | Technical reference: [`docs/scoring_methodology.md`](scoring_methodology.md)

## What Could Be Done Next

1. **Dashboard polish** — Export buttons, custom theming, additional interactivity.
2. **PDF report export** — `report_generator.py` outputs Markdown; could add pandoc-based PDF generation.
3. **Benchmark evaluation** — Compare rule-based vs Qwen classifications on a labeled test set to quantify LLM value-add.
4. **Additional data sources** — Integrate MRA/MRIA supervisory documents (some are in `manual_sources/`).
5. **Presentation dry-run** — The 13-slide HTML presentation is ready but hasn't been rehearsed with live dashboard demo.

## How to Run

```bash
cd /Users/adityadabeer/Documents/VSCODE/misc/bufn403/BUFN403_Capstone

# Run tests
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
- `QwenAnswerGenerator` (`src/ai_corpus/qwen.py`) handles LLM inference with local/remote fallback and partial-JSON recovery
- `classification_io.py` handles all classification I/O: JSONL read/write, normalization, confidence parsing, malformed-tail repair for resume
- `has_ai_anchor()` (`src/ai_corpus/pipeline.py`) is the shared AI keyword filter
- CLI uses argparse subcommands in `scripts/ai_corpus_pipeline.py`
- Dashboard reads canonical artifacts only — no mock data or heuristic fallback in the default path
- Qwen classification is designed to run on Google Colab with GPU; the pipeline supports resume across interrupted sessions
