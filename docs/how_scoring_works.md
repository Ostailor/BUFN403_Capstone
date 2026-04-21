# How the Scoring Works

A complete explanation of how every bank in this project gets scored, ranked, and compared. This document covers the full pipeline from raw text to final composite ranking.

---

## Table of Contents

1. [The Big Picture](#1-the-big-picture)
2. [Step 1: Filtering for AI-Relevant Text](#2-step-1-filtering-for-ai-relevant-text)
3. [Step 2: Intent Level Classification](#3-step-2-intent-level-classification)
4. [Step 3: Application Category Tagging](#4-step-3-application-category-tagging)
5. [Step 4: Maturity Score](#5-step-4-maturity-score)
6. [Step 5: Breadth Score](#6-step-5-breadth-score)
7. [Step 6: Momentum Score](#7-step-6-momentum-score)
8. [Step 7: Composite Score and Final Ranking](#8-step-7-composite-score-and-final-ranking)
9. [Worked Example](#9-worked-example)

---

## 1. The Big Picture

We analyze 50 U.S. banks to understand how seriously each one is adopting AI. The raw inputs are public text documents — earnings call transcripts, SEC filings (10-K, 10-Q, 8-K, DEF14A), and FDIC call reports — spanning 8 quarters (2024 Q1 through 2025 Q4).

These documents are split into ~256,000 text chunks. From those, we identify the ones that actually talk about AI (~2,757 chunks), classify each one along two dimensions (intent level and application category), then aggregate those classifications into three component scores per bank. Those three scores combine into a single Composite Score that determines the final ranking.

The flow looks like this:

```
Raw Documents
  → ~256K text chunks
  → ~2,757 AI-relevant chunks (filtered by anchor patterns)
  → Each chunk classified:
      - Intent Level (1-4)
      - Application Categories (1-6 tags)
  → Per-bank aggregation:
      - Maturity Score (0-100)
      - Breadth Score (0-100)
      - Momentum Score (-100 to +100)
  → Composite Score (0-100)
  → Final Ranking (1-50)
```

---

## 2. Step 1: Filtering for AI-Relevant Text

Not every chunk of text from a bank's filings is about AI. Most of it is about interest rates, loan portfolios, regulatory capital, etc. We use 9 regex anchor patterns to identify chunks that are actually discussing artificial intelligence or related technology. Only chunks that match at least one anchor pattern proceed to classification.

This filtering step takes us from ~256,000 total chunks down to ~2,757 AI-relevant chunks.

---

## 3. Step 2: Intent Level Classification

Each AI-relevant chunk is assigned an **intent level** from 1 to 4. The intent level captures *how far along* a bank is with the AI initiative described in that chunk — are they just thinking about it, or have they already rolled it out across the entire organization?

| Level | Label | What It Means | Example Signal Words |
|-------|-------|---------------|----------------------|
| 1 | **Exploring** | The bank is investigating, researching, or piloting AI in this area | investigating, considering, evaluating, studying, piloting, researching, exploring, assessing |
| 2 | **Committing** | The bank has committed resources — budget, headcount, partnerships — to build something | investing, budgeting, building, planning, allocating, developing, committed, funding |
| 3 | **Deploying** | The AI system is live and in use | launched, implemented, using, rolled out, live, operational, deployed, in production |
| 4 | **Scaling** | The AI system is being expanded across the organization | expanding, enterprise-wide, firm-wide, increasing, doubling down, across all, scaled, scaling |

### How classification works

Classification uses a **Qwen LLM** (run on Google Colab with GPU). The model receives each chunk along with the intent level definitions and application categories, and returns a structured JSON response with the intent level, application categories, confidence score, and an evidence snippet.

There is also a **rule-based fallback** path that uses signal word matching. It checks the chunk text against the signal words for each level, starting from the highest level (4) and working down. The first match wins. However, the primary pipeline uses Qwen, and the rule-based path exists only as a reference — it does not silently substitute for Qwen when the model fails. If Qwen fails, the pipeline fails loudly.

### Confidence scores

Each classification includes a confidence score. Rule-based classifications always get confidence = 0.5. Qwen-based classifications return a model-estimated confidence that can vary. The confidence score is tracked but does not currently affect the downstream scoring.

---

## 4. Step 3: Application Category Tagging

In addition to intent level, each chunk is tagged with one or more **application categories** describing *what kind* of AI the bank is talking about. There are exactly 6 categories:

| Category | What It Covers | Example Keywords |
|----------|---------------|------------------|
| **GenAI / LLMs** | Chatbots, copilots, generative AI, large language models | chatbot, copilot, generative ai, llm, gpt, prompt, genai |
| **Predictive ML** | Credit scoring, forecasting, propensity models | credit scoring, forecasting, regression, prediction, predictive |
| **NLP / Text** | Document processing, sentiment analysis, text mining | document processing, sentiment analysis, entity extraction, nlp |
| **Computer Vision** | Check imaging, ID verification, facial recognition, OCR | check imaging, id verification, facial recognition, ocr, computer vision |
| **RPA / Automation** | Process automation, workflow automation | process automation, straight-through processing, workflow automation, rpa |
| **Fraud / Risk Models** | Fraud detection, AML, transaction monitoring, anomaly detection | fraud detection, aml, transaction monitoring, anomaly detection, risk model |

A single chunk can be tagged with multiple categories. For example, a chunk about "using generative AI to improve our fraud detection capabilities" would be tagged with both **GenAI / LLMs** and **Fraud / Risk Models**.

---

## 5. Step 4: Maturity Score

**Range:** 0 to 100

**What it measures:** How far along the explore-to-scale spectrum a bank's AI initiatives are, on average.

### Formula

```
Maturity = (average intent level across all of a bank's classified chunks) / 4 × 100
```

Equivalently:

```
Maturity = sum(all intent levels) / count(chunks) × 25
```

### Intuition

- A bank where every chunk is level 1 (Exploring) scores **25.0** — they're talking about AI but haven't done anything yet.
- A bank where every chunk is level 4 (Scaling) scores **100.0** — everything they mention is already deployed at enterprise scale.
- Most banks fall somewhere in between. A mix of Exploring, Committing, and Deploying chunks might average around 2.5, giving a maturity of **62.5**.

### Examples

| Scenario | Intent Levels | Average | Maturity |
|----------|---------------|---------|----------|
| All exploring | [1, 1, 1, 1] | 1.0 | 25.0 |
| Mixed early stage | [1, 1, 2, 2, 1] | 1.4 | 35.0 |
| Balanced spread | [1, 2, 3, 4] | 2.5 | 62.5 |
| Mostly deploying | [3, 3, 3, 4, 2] | 3.0 | 75.0 |
| All scaling | [4, 4, 4] | 4.0 | 100.0 |

---

## 6. Step 5: Breadth Score

**Range:** 0 to 100

**What it measures:** How diversified a bank's AI applications are across the 6 application categories. A bank that only does fraud detection scores low; a bank that applies AI across fraud, GenAI, automation, NLP, ML, and computer vision scores high.

### Formula (Shannon Entropy, normalized)

First, count how many AI mentions a bank has in each of the 6 categories. Convert those counts into proportions:

```
p_i = mentions in category i / total mentions across all categories
```

Then compute Shannon entropy:

```
H = -Σ (p_i × log₂(p_i))    for each category where p_i > 0
```

Normalize by the maximum possible entropy (which occurs when mentions are perfectly evenly distributed across all 6 categories):

```
H_max = log₂(6) ≈ 2.585

Breadth = (H / H_max) × 100
```

### Intuition

Shannon entropy measures how "spread out" or "balanced" a distribution is.

- **Maximum entropy** (Breadth = 100): The bank has equal mentions in all 6 categories. Their AI strategy is maximally diversified.
- **Zero entropy** (Breadth = 0): All of the bank's AI mentions fall in a single category. They're a one-trick pony.
- **In between**: The more categories a bank covers, and the more evenly distributed its mentions are, the higher the score. Having 5 categories but 90% in one of them still scores lower than having 4 categories with equal distribution.

### Examples

| Distribution | Entropy (H) | Breadth |
|-------------|-------------|---------|
| Equal across all 6 categories (16.7% each) | 2.585 | 100.0 |
| Equal across 3 categories, 0 in other 3 | 1.585 | 61.3 |
| Two categories: 50/50 split | 1.000 | 38.7 |
| Two categories: 90/10 split | 0.469 | 18.1 |
| 100% in one category | 0.000 | 0.0 |

### Why Shannon entropy?

Simple alternatives like "count of categories used" don't capture how concentrated or balanced the distribution is. A bank with 95% fraud detection and 5% GenAI would score the same as one with 50/50 — but those are very different strategies. Shannon entropy naturally penalizes concentration and rewards balance, which is exactly what we want for measuring breadth of AI adoption.

---

## 7. Step 6: Momentum Score

**Range:** -100 to +100

**What it measures:** Whether a bank's AI maturity is accelerating (positive) or decelerating (negative) over time. It captures the *trajectory* rather than the current state.

### Formula

First, compute a **per-quarter maturity score** for each bank. This is the same maturity formula from Step 4, but applied only to chunks from a specific quarter:

```
Quarterly Maturity(Q) = average intent level of chunks in quarter Q / 4 × 100
```

Then fit an **ordinary least squares (OLS) linear regression** through the quarterly maturity scores over time:

```
slope = OLS regression slope of quarterly maturity scores
```

The slope represents how many maturity points the bank gains (or loses) per quarter. Multiply by 4 to **annualize** it (maturity-point change per year), then clamp to the [-100, +100] range:

```
Momentum = clamp(slope × 4, -100, +100)
```

### Intuition

- **Positive momentum** (+): The bank's AI maturity is increasing over time. They're moving up the explore→scale spectrum.
- **Negative momentum** (−): The bank's AI maturity is decreasing. Maybe they talked big early and pulled back, or early-stage mentions were more optimistic than later ones.
- **Near zero**: Stable trajectory — the bank's maturity level isn't changing much quarter to quarter.

### Why OLS regression?

A simple "last quarter minus first quarter" approach is fragile — one unusual quarter throws it off. OLS regression fits a trend line through all available quarters, which smooths out quarterly noise and gives a more robust estimate of the direction.

### Examples

| Quarterly Maturity Scores | Slope | Annualized | Momentum |
|---------------------------|-------|------------|----------|
| [40, 45, 50, 55] | +5.0 per quarter | +20.0 per year | **+20.0** |
| [60, 60, 60, 60] | 0.0 | 0.0 | **0.0** |
| [70, 65, 55, 50] | −6.67 per quarter | −26.7 per year | **−26.7** |
| [30, 35, 40, 45, 50, 55, 60, 65] | +5.0 per quarter | +20.0 per year | **+20.0** |

### Edge cases

- If a bank has data for only 1 quarter, momentum defaults to **0.0** (you can't compute a slope from a single point).
- If a bank has data for exactly 2 quarters, the slope is just the difference between them (no smoothing benefit, but still valid).

---

## 8. Step 7: Composite Score and Final Ranking

**Range:** 0 to 100

**What it measures:** A single number that combines all three dimensions — maturity, breadth, and momentum — into one ranking metric.

### Formula

Because momentum ranges from -100 to +100 while the other scores range from 0 to 100, we first normalize momentum to a 0-100 scale:

```
Momentum_normalized = (Momentum + 100) / 2
```

This maps:
- Momentum of -100 → 0
- Momentum of 0 → 50
- Momentum of +100 → 100

Then combine with weights:

```
Composite = 0.50 × Maturity + 0.35 × Breadth + 0.15 × Momentum_normalized
```

### Why these weights?

| Component | Weight | Rationale |
|-----------|--------|-----------|
| **Maturity** | 50% | The most important signal. It directly measures how advanced a bank's AI use actually is. A bank that has deployed and scaled AI is doing more than one that's just exploring. |
| **Breadth** | 35% | The second most important signal. A bank applying AI across many domains (fraud, automation, GenAI, etc.) has deeper organizational commitment than one focused on a single use case. |
| **Momentum** | 15% | The least weighted component. It's the noisiest signal because it depends on quarter-to-quarter variation in public disclosures, which can be influenced by what questions analysts happen to ask on an earnings call. It's valuable directional context but shouldn't dominate the ranking. |

### Ranking

After computing the Composite Score for all 50 banks, they are sorted in descending order. The bank with the highest Composite Score gets Rank 1.

### Interactive weight overrides (AI Classification dashboard only)

The AI Classification & Intent pages let a viewer override the default Composite weights live via a sidebar widget labelled **AI Classification — Composite Weights**. This is a view-only override:

- The stored artifact (`bank_composite_scores.csv`) still uses the canonical 0.50 / 0.35 / 0.15 weights. Nothing on disk changes when you move a slider.
- The widget exposes two integer-percent sliders — **Maturity** and **Breadth** — and shows **Momentum** as a derived metric equal to `100 − Maturity − Breadth`. The three weights therefore always sum to exactly 100% by construction; no manual balancing is required.
- Moving Maturity up automatically tightens Breadth's allowed range; Momentum is recomputed on every change.
- The **Reset to defaults** button restores 50 / 35 / 15 in one click (implemented via a Streamlit `on_click` callback so the reset runs before widgets instantiate).
- Every time the weights change, `Composite` is recomputed row-by-row and `Rank` is re-sorted (ties broken by ticker).
- Weights persist in `st.session_state`, so navigating between Leaderboard, Bank Deep Dive, Compare Banks, and Market Overview keeps the same weighting.

Use this when you want to ask "what if Momentum mattered more?" without re-running the pipeline.

---

## 9. Worked Example

Let's walk through a complete scoring for a hypothetical bank, **ACME Corp**.

### Classification results

ACME has 20 AI-relevant chunks across 4 quarters:

- **2024 Q3** (5 chunks): intent levels [1, 1, 2, 1, 2], categories mostly GenAI and Fraud
- **2024 Q4** (5 chunks): intent levels [2, 2, 3, 2, 2], categories GenAI, Fraud, NLP
- **2025 Q1** (5 chunks): intent levels [2, 3, 3, 3, 2], categories GenAI, Fraud, NLP, RPA
- **2025 Q2** (5 chunks): intent levels [3, 3, 4, 3, 3], categories GenAI, Fraud, NLP, RPA, Predictive ML

### Maturity

Average intent level across all 20 chunks:

```
avg = (1+1+2+1+2 + 2+2+3+2+2 + 2+3+3+3+2 + 3+3+4+3+3) / 20
    = 47 / 20
    = 2.35

Maturity = 2.35 / 4 × 100 = 58.75
```

### Breadth

Count mentions across categories (a chunk can appear in multiple categories):

| Category | Mentions | Proportion |
|----------|----------|------------|
| GenAI / LLMs | 14 | 0.311 |
| Fraud / Risk Models | 12 | 0.267 |
| NLP / Text | 9 | 0.200 |
| RPA / Automation | 6 | 0.133 |
| Predictive ML | 4 | 0.089 |
| Computer Vision | 0 | 0.000 |

Shannon entropy (only categories with p > 0):

```
H = -(0.311×log₂(0.311) + 0.267×log₂(0.267) + 0.200×log₂(0.200) + 0.133×log₂(0.133) + 0.089×log₂(0.089))
  = -(0.311×(-1.685) + 0.267×(-1.906) + 0.200×(-2.322) + 0.133×(-2.910) + 0.089×(-3.490))
  = -(−0.524 + −0.509 + −0.464 + −0.387 + −0.311)
  = 2.195

H_max = log₂(6) = 2.585

Breadth = (2.195 / 2.585) × 100 = 84.91
```

ACME covers 5 of 6 categories with a reasonably balanced distribution, so they score high on breadth.

### Momentum

Per-quarter maturity scores:

```
2024 Q3: avg([1,1,2,1,2]) = 1.4 → Maturity = 35.0
2024 Q4: avg([2,2,3,2,2]) = 2.2 → Maturity = 55.0
2025 Q1: avg([2,3,3,3,2]) = 2.6 → Maturity = 65.0
2025 Q2: avg([3,3,4,3,3]) = 3.2 → Maturity = 80.0
```

OLS regression through [35.0, 55.0, 65.0, 80.0]:

```
x values: [0, 1, 2, 3], y values: [35.0, 55.0, 65.0, 80.0]
x̄ = 1.5, ȳ = 58.75
slope = Σ((x-x̄)(y-ȳ)) / Σ((x-x̄)²) = 67.5 / 5 = 13.5

Momentum = clamp(13.5 × 4, -100, 100) = clamp(54.0, -100, 100) = 54.0
```

ACME is accelerating fast — gaining 13.5 maturity points per quarter.

### Composite

```
Momentum_normalized = (54.0 + 100) / 2 = 77.0
Composite = 0.50 × 58.75 + 0.35 × 84.91 + 0.15 × 77.0
          = 29.375 + 29.719 + 11.55
          = 70.64
```

ACME scores **70.64**, which in the actual dataset would place them in the top 10.
