# AI Intent & Classification Ranking System — Design Spec

## Context

BUFN403 Capstone project analyzes AI adoption across 50 major U.S. banks. The team's subtopic is **Classification and Intent**. The existing pipeline scores banks on a flat 1-10 scale with keyword-based theme tagging — useful but one-dimensional. This feature adds a **two-dimensional classification system** (intent level + application type) with composite scoring, an interactive Streamlit dashboard, and a static report. The goal: wow the teacher and project sponsors with a system that tells a *story* about how banks are actually adopting AI — not just who mentions it most.

**Timeline:** 1 week  
**LLM engine:** Qwen via HuggingFace API (already integrated)  
**Bank scope:** All banks with available corpus data  

---

## 1. Classification Engine

### 1.1 Intent Classification (4 levels)

Each AI-related chunk gets classified by the **intent** behind the mention:

| Level | Label        | Signal Words |
|-------|-------------|-------------|
| 1     | Exploring    | investigating, considering, evaluating, studying, piloting, researching |
| 2     | Committing   | investing, budgeting, building, planning, allocating, developing |
| 3     | Deploying    | launched, implemented, using, rolled out, live, operational |
| 4     | Scaling      | expanding, enterprise-wide, increasing, doubling down, across all, firm-wide |

### 1.2 Application Taxonomy (6 categories)

Each chunk gets tagged with one or more **application categories**:

| Category            | Example Keywords |
|--------------------|-----------------|
| GenAI / LLMs       | chatbot, copilot, generative AI, large language model, GPT, prompt |
| Predictive ML      | credit scoring, forecasting, propensity model, regression, prediction |
| NLP / Text         | document processing, sentiment analysis, entity extraction, text mining |
| Computer Vision    | check imaging, ID verification, facial recognition, OCR |
| RPA / Automation   | process automation, straight-through processing, workflow automation, robotic |
| Fraud / Risk Models| fraud detection, AML, transaction monitoring, anomaly detection, risk model |

### 1.3 Classification Pipeline

**Input:** Existing `artifacts/ai_corpus/chunks.jsonl` (chunks already filtered for AI keywords)

**Process:**
1. Load chunks from JSONL
2. Batch chunks (e.g., 10 at a time) to Qwen via HF API
3. Prompt returns structured JSON per chunk:
   ```json
   {
     "chunk_id": "BAC__10k_2024q4__chunk_0012",
     "intent_level": 3,
     "intent_label": "Deploying",
     "app_categories": ["GenAI / LLMs", "Fraud / Risk Models"],
     "confidence": 0.85,
     "evidence_snippet": "We have deployed generative AI tools across our fraud operations..."
   }
   ```
4. Fallback: If Qwen fails, use rule-based keyword matching as a deterministic backup

**Output:** `artifacts/ai_corpus/classifications.jsonl`

**New file:** `src/ai_corpus/intent_classifier.py`

---

## 2. Composite Scoring & Ranking

### 2.1 Three Component Scores

**AI Maturity Score (0-100):**
- Weighted sum of intent levels per bank
- `score = sum(level_weight × count_at_level) / total_mentions × 25`
- Weights: Exploring=1, Committing=2, Deploying=3, Scaling=4
- Normalized so volume alone doesn't dominate

**AI Breadth Score (0-100):**
- Coverage across the 6 application categories
- `base = (categories_with_mentions / 6) × 80`
- `depth_bonus = min(20, categories_with_3plus_mentions × 5)`
- `score = base + depth_bonus`

**AI Momentum Score (-100 to +100):**
- Quarter-over-quarter change in maturity score
- `momentum = maturity_current_half - maturity_prior_half`
- Positive = accelerating intent, Negative = regression/silence
- Requires at least 2 quarters of data

### 2.2 Composite Rank

```
composite = 0.45 × maturity + 0.30 × breadth + 0.25 × momentum_normalized
```

Where `momentum_normalized = (momentum + 100) / 2` to map [-100, 100] → [0, 100]

### 2.3 Output Artifacts

| File | Contents |
|------|----------|
| `bank_composite_scores.csv` | Per-bank: maturity, breadth, momentum, composite, rank |
| `quarterly_progression.csv` | Per-bank per-quarter: maturity score, intent distribution |
| `app_category_matrix.csv` | Per-bank per-category: mention count, avg intent level |

**New file:** `src/ai_corpus/composite_scorer.py`

---

## 3. Streamlit Dashboard

### 3.1 Page 1 — Leaderboard (`pages/1_Leaderboard.py`)
- Ranked table: Bank, Composite Score, Maturity, Breadth, Momentum, Rank
- Color-coded cells (green→red gradient)
- Sort by any column
- Filter by score range, application category
- Click bank name to navigate to deep dive

### 3.2 Page 2 — Bank Deep Dive (`pages/2_Bank_Deep_Dive.py`)
- Bank selector dropdown
- **Intent Distribution** — Donut chart (% at each intent level)
- **Application Radar** — 6-axis radar chart (coverage per category)
- **Quarterly Timeline** — Line chart showing maturity score progression over quarters
- **Evidence Table** — Top 10 classified chunks with source, intent label, categories, and actual text snippet

### 3.3 Page 3 — Comparative Analysis (`pages/3_Compare_Banks.py`)
- Multi-select: choose 2-4 banks
- Overlaid radar charts
- Side-by-side intent distribution bars
- Comparative metrics table highlighting where each bank leads

### 3.4 Page 4 — Market Overview (`pages/4_Market_Overview.py`)
- **Heatmap** — Banks × Application categories (color by mention count)
- **Industry Intent Stack** — Stacked bar chart of intent distribution across all banks
- **Maturity Quadrant** — Scatter: x=Total Assets, y=Composite Score (who punches above weight?)
- **Top Movers** — Bar chart of banks with highest/lowest momentum

### 3.5 App Entry Point

**New file:** `dashboard/app.py` (Streamlit main)  
**New directory:** `dashboard/pages/` (multi-page Streamlit layout)

Dependencies to add: `streamlit`, `plotly` (for interactive charts)

---

## 4. Static Report Generator

Auto-generated Markdown report with embedded chart images:

**Sections:**
1. Executive Summary — Top 5 banks, key findings
2. Methodology — How classification and scoring works
3. Full Rankings Table
4. Key Visualizations — Heatmap, quadrant, intent distribution (saved as PNGs via matplotlib)
5. Bank Highlights — Top 3 banks with detailed profiles
6. Trends & Momentum — Quarter-over-quarter insights

**Output:** `artifacts/ai_corpus/ai_intent_classification_report.md`  
**New file:** `src/ai_corpus/report_generator.py`

---

## 5. Integration with Existing Codebase

### Files to create:
| File | Purpose |
|------|---------|
| `src/ai_corpus/intent_classifier.py` | Classification engine (Qwen prompt + rule-based fallback) |
| `src/ai_corpus/composite_scorer.py` | Scoring and ranking logic |
| `src/ai_corpus/report_generator.py` | Static Markdown report generation |
| `dashboard/app.py` | Streamlit entry point |
| `dashboard/pages/1_Leaderboard.py` | Leaderboard page |
| `dashboard/pages/2_Bank_Deep_Dive.py` | Bank detail page |
| `dashboard/pages/3_Compare_Banks.py` | Comparison page |
| `dashboard/pages/4_Market_Overview.py` | Market overview page |
| `tests/test_intent_classifier.py` | Classification tests |
| `tests/test_composite_scorer.py` | Scoring tests |

### Files to modify:
| File | Change |
|------|--------|
| `src/ai_corpus/__init__.py` | Export new modules |
| `src/ai_corpus/config.py` | Add intent levels, app categories, classifier config |
| `scripts/ai_corpus_pipeline.py` | Add CLI subcommands: `classify`, `score`, `report`, `dashboard` |
| `requirements.txt` | Add `streamlit`, `plotly` |

### Existing code to reuse:
| What | Where |
|------|-------|
| Chunk loading & AI-anchor filtering | `src/ai_corpus/pipeline.py` — `normalize_corpus()` |
| Qwen inference (remote + local) | `src/ai_corpus/qwen.py` — `QwenAnswerGenerator` |
| Theme keyword patterns | `src/ai_corpus/config.py` — `THEME_KEYWORDS` |
| Bank roster loading | `src/ai_corpus/config.py` — `BANK_ROSTER_PATH` |
| Existing scoring data | `AI_Bank_Classification.csv` |
| Matplotlib plot patterns | `scripts/plot_bank_ai_results.py` |

---

## 6. Parallel Agent Execution Plan

### Wave 1 (parallel — no dependencies):
- **Agent 1: Classifier** — Build `intent_classifier.py` with Qwen prompt engineering, rule-based fallback, batch processing
- **Agent 3: Dashboard** — Build full Streamlit app with mock/sample data (hardcoded CSVs mirroring expected schema)
- **Agent 5: Tests** — Write tests for classifier and scorer using fixture data

### Wave 2 (after Agent 1 completes):
- **Agent 2: Scorer** — Build `composite_scorer.py`, reads classifications, outputs all score CSVs

### Wave 3 (after Agent 2 completes):
- **Agent 4: Report** — Build `report_generator.py`, generates Markdown with matplotlib charts
- **Agent 3 (continued):** Wire dashboard to real CSV outputs, remove mock data

### Wave 4 (final):
- **Integration & CLI** — Add subcommands to `scripts/ai_corpus_pipeline.py`, update `__init__.py`, run full pipeline end-to-end

---

## 7. Verification Plan

### Unit tests:
- `test_intent_classifier.py` — Mock Qwen responses, verify parsing, test rule-based fallback
- `test_composite_scorer.py` — Known inputs → expected scores, edge cases (single mention, missing quarters)

### Integration test:
- Run classifier on a small subset of real chunks → verify output schema
- Run scorer on classifier output → verify CSV schemas and score ranges
- Run report generator → verify Markdown output exists and contains expected sections

### Dashboard test:
- `streamlit run dashboard/app.py` — verify all 4 pages load without errors
- Verify charts render with real data
- Test bank selector, filters, and comparison features

### End-to-end:
```bash
python scripts/ai_corpus_pipeline.py classify
python scripts/ai_corpus_pipeline.py score
python scripts/ai_corpus_pipeline.py report
streamlit run dashboard/app.py
```
