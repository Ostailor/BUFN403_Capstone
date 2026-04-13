# Scoring Methodology

Reference documentation for the AI adoption scoring system used in the BUFN403 Capstone project.

## 1. Overview

The system scores 50 U.S. banks on AI adoption using earnings transcripts, SEC filings, and FDIC call reports. Text chunks are classified by intent level and application category, then aggregated into three component scores (Maturity, Breadth, Momentum) which combine into a single Composite Score for ranking.

## 2. Classification

### AI Chunk Filtering

From ~256K total text chunks, AI-relevant chunks are identified using 9 AI-anchor regex patterns. This yields ~2,757 classified chunks.

### Intent Level Assignment

Each AI chunk is classified into one of four intent levels via signal word matching (highest level matched first):

| Level | Label | Example Signals |
|-------|-------|-----------------|
| 1 | Exploring | investigating, evaluating, piloting, considering |
| 2 | Committing | investing, building, developing, partnering |
| 3 | Deploying | launched, implemented, operational, live |
| 4 | Scaling | enterprise-wide, expanding, scaling, across all |

### Application Category Tagging

Each chunk is tagged with one or more of 6 application categories via keyword matching:

- GenAI / LLMs
- Predictive ML
- NLP / Text Processing
- Computer Vision
- RPA / Automation
- Fraud / Risk Models

### Confidence

Rule-based classification assigns confidence = 0.5 for all chunks (vs. variable confidence for LLM-based classification).

## 3. Maturity Score (0-100)

Measures how far along a bank's AI initiatives are on the explore-to-scale spectrum.

**Formula:**

```
maturity = avg(intent_levels) / 4 * 100
         = sum(intent_levels) / count * 25
```

Intent levels are linearly weighted: Scaling (4) is 4x more valuable than Exploring (1).

**Examples:**

| Bank Chunks | Intent Levels | Maturity |
|-------------|---------------|----------|
| 4 chunks | [1, 2, 3, 4] | 2.5 / 4 * 100 = 62.5 |
| 3 chunks | [4, 4, 4] | 4.0 / 4 * 100 = 100.0 |
| 5 chunks | [1, 1, 1, 2, 1] | 1.2 / 4 * 100 = 30.0 |

## 4. Breadth Score (0-100) -- Shannon Entropy

Measures how diversified a bank's AI applications are across the 6 categories.

**Formula (Shannon entropy, normalized):**

```
H = -sum(p_i * log2(p_i))   for each of 6 categories where p_i > 0
H_max = log2(6) ≈ 2.585
breadth = (H / H_max) * 100
```

Where `p_i` is the proportion of a bank's AI mentions in category `i`.

Shannon entropy rewards balance across categories and penalizes concentration. A bank with all mentions in one category scores 0; a bank with equal distribution across all 6 scores 100.

**Examples:**

| Distribution | H | Breadth |
|-------------|---|---------|
| Equal across 6 categories | 2.585 | 100.0 |
| Equal across 3 categories, 0 in other 3 | 1.585 | 61.3 |
| 100% in one category | 0.0 | 0.0 |
| 90% GenAI, 10% RPA | 0.469 | 18.1 |

## 5. Momentum Score (-100 to +100) -- Linear Regression Slope

Measures whether a bank's AI adoption is accelerating or decelerating over time.

**Formula:**

```
slope = OLS regression slope of quarterly maturity scores over time
momentum = clamp(slope * 4, -100, 100)
```

The slope represents maturity points gained per quarter. Multiplying by 4 annualizes it (maturity-point change per year). The result is clamped to [-100, +100].

- Positive momentum = accelerating AI adoption
- Negative momentum = decelerating AI adoption
- Near zero = stable trajectory

**Examples:**

| Quarterly Maturity Scores | Slope | Momentum |
|---------------------------|-------|----------|
| [40, 45, 50, 55] | +5.0 | clamp(20.0, -100, 100) = +20.0 |
| [60, 60, 60, 60] | 0.0 | 0.0 |
| [70, 65, 55, 50] | -6.67 | clamp(-26.7, -100, 100) = -26.7 |

## 6. Composite Score (0-100)

Combines the three component scores into a single ranking metric.

**Formula:**

```
momentum_normalized = (momentum + 100) / 2    # maps [-100, +100] → [0, 100]
composite = 0.50 * maturity + 0.35 * breadth + 0.15 * momentum_normalized
```

**Weight rationale:**

| Component | Weight | Rationale |
|-----------|--------|-----------|
| Maturity | 0.50 | Most important -- directly measures how advanced a bank's AI use is |
| Breadth | 0.35 | Second -- diversified AI strategy indicates organizational depth |
| Momentum | 0.15 | Least weight -- noisiest signal, sensitive to quarterly variation |

**Example:**

A bank with maturity=60, breadth=80, momentum=+20:

```
momentum_normalized = (20 + 100) / 2 = 60
composite = 0.50 * 60 + 0.35 * 80 + 0.15 * 60 = 30 + 28 + 9 = 67.0
```
