# BUFN403 Capstone — AI Intent & Classification Ranking System
## Planning Session Transcript

**Date:** April 13, 2026
**Repo:** https://github.com/Ostailor/BUFN403_Capstone
**Local path:** `/Users/adityadabeer/Documents/VSCODE/misc/bufn403/BUFN403_Capstone`

---

## Project Overview

**BUFN403 Capstone** is a production-grade AI usage corpus audit and RAG pipeline by Om Tailor that analyzes how 50 major U.S. banks discuss AI across earnings transcripts, SEC filings, FDIC call reports, and supervisory documents.

### Key highlights of existing codebase
- Two pipelines — a classic rule+LLM hybrid scorer and a modern RAG system with Chroma + DuckDB + Qwen
- 9 AI themes tracked (strategy, use cases, investment, risk, governance, etc.)
- Local-first design — Qwen models with fallback chain, no cloud dependency required
- Full analysis suite — scorecards, heatmaps, quadrant plots, per-bank summaries, topic findings
- ~3,000+ lines of core pipeline code with tests

---

## Design Decisions (Brainstorming Q&A)

| Question | Answer |
|----------|--------|
| Classification and Intent scope | Both intent + application type |
| Output format | Both dashboard + report |
| LLM approach | Qwen via HuggingFace API (Recommended — free, already wired up) |
| Bank coverage | Whatever data is available |
| Timeline | 1 week |
| Approach selected | Approach A: Intent + Classification Matrix |

---

## Section 1: Classification Engine

Qwen-powered classifier reads each text chunk from the existing corpus and outputs structured labels.

### Intent Classification (4 levels)

| Level | Label | Signal Words |
|-------|-------|--------------|
| 1 | Exploring | investigating, considering, evaluating, studying, piloting |
| 2 | Committing | investing, budgeting, building, planning, allocating |
| 3 | Deploying | launched, implemented, using, rolled out, live |
| 4 | Scaling | expanding, enterprise-wide, increasing, doubling down, across all |

### Application Taxonomy (6 categories)

| Category | Examples |
|----------|----------|
| GenAI / LLMs | chatbots, copilots, generative AI, large language models |
| Predictive ML | credit scoring, forecasting, propensity models |
| NLP / Text | document processing, sentiment analysis, entity extraction |
| Computer Vision | check imaging, ID verification, facial recognition |
| RPA / Automation | process automation, straight-through processing |
| Fraud / Risk Models | fraud detection, AML, transaction monitoring |

### How it works
1. Read existing `chunks.jsonl` from the corpus
2. For each chunk containing AI keywords, send to Qwen with a structured prompt
3. Qwen returns JSON: `{intent_level, intent_label, app_categories: [], confidence}`
4. Store results in a new `classifications.jsonl` artifact

---

## Section 2: Composite Scoring & Ranking

Each bank gets 3 composite scores that combine into a final rank.

### 1. AI Maturity Score (0-100)
- Weighted sum of intent levels: Exploring=1, Committing=2, Deploying=3, Scaling=4
- Normalized by total mention count (so big banks don't auto-win by volume)
- **Formula:** `sum(intent_weight × mention_count) / total_mentions × 25`

### 2. AI Breadth Score (0-100)
- How many of the 6 application categories does the bank cover?
- Weighted by depth: `(categories_used / 6) × 100`, with a bonus for categories with 3+ mentions

### 3. AI Momentum Score (-100 to +100)
- Quarter-over-quarter change in maturity score
- Positive = intent is progressing (exploring → deploying), Negative = regression or silence
- Captures "who's accelerating" — great for storytelling

### Final Composite Rank
```
0.45 × Maturity + 0.30 × Breadth + 0.25 × Momentum
```
Weights emphasize maturity (what matters most) but reward breadth and momentum.

### Output artifacts
- `ai_intent_classifications.csv` — per-chunk classifications
- `bank_composite_scores.csv` — per-bank scores and ranks
- `quarterly_progression.csv` — time-series data for momentum

---

## Section 3: Streamlit Dashboard

Multi-page Streamlit app with these views:

### Page 1 — Leaderboard
- Ranked table of all banks with Maturity, Breadth, Momentum, and Composite scores
- Color-coded cells (green = high, red = low)
- Click any bank to drill into its detail page
- Filters: by score range, by application category

### Page 2 — Bank Deep Dive
Select a bank → see its:
- Intent distribution pie chart (% Exploring/Committing/Deploying/Scaling)
- Application category radar chart (coverage across 6 types)
- Quarterly intent progression timeline (line chart showing maturity over time)
- Top evidence snippets (actual quotes from filings/transcripts with source citations)

### Page 3 — Comparative Analysis
Side-by-side comparison of 2-4 banks:
- Overlaid radar charts
- Intent progression comparison
- "Where Bank A leads vs Bank B" summary

### Page 4 — Market Overview
- Bank × Application heatmap (all banks, all categories)
- Intent distribution across the industry (stacked bar)
- AI Maturity Quadrant: Assets vs Composite Score scatter plot (who's punching above their weight?)
- Top movers: banks with highest momentum scores

### Static Report
- Auto-generated Markdown/PDF with all key charts and a narrative summary
- Exportable for the leave-behind deliverable

---

## Section 4: Parallel Agent Strategy

To hit the 1-week timeline, the implementation will be split across independent parallel agents.

| Agent | Scope | Dependencies |
|-------|-------|--------------|
| Agent 1: Classifier | Build Qwen classification prompt, `intent_classifier.py`, run on all chunks, output `classifications.jsonl` | Existing `chunks.jsonl` |
| Agent 2: Scorer | Build `composite_scorer.py` — reads classifications, computes Maturity/Breadth/Momentum, outputs CSVs | Agent 1's output (runs after) |
| Agent 3: Dashboard | Build Streamlit app with all 4 pages, using mock data first, then swap in real CSVs | Can start with mock data in parallel with Agents 1-2 |
| Agent 4: Static Report | Build auto-generated Markdown report with embedded charts | Agent 2's output (runs after) |
| Agent 5: Tests | Write pytest tests for classifier, scorer, and integration | Can run in parallel with Agent 3 |

### Parallelism Waves
- **Wave 1 (parallel):** Agent 1 (Classifier) + Agent 3 (Dashboard with mock data) + Agent 5 (Tests)
- **Wave 2 (after Agent 1):** Agent 2 (Scorer)
- **Wave 3 (after Agent 2):** Agent 4 (Report) + Agent 3 gets real data wired in

This maximizes throughput while respecting data dependencies.

---

## Spec File Location

The full design spec was written to:
```
BUFN403_Capstone/docs/superpowers/specs/2026-04-13-ai-intent-classification-ranking-design.md
```
(238 lines)

### Self-Review Findings
1. No TBDs or TODOs — all sections are concrete
2. Internal consistency checks: scoring formula maps correctly to intent levels, dashboard pages reference the right CSVs
3. Scope is focused — single feature, clear boundaries
4. Minor fix needed: specify the Qwen prompt template structure
