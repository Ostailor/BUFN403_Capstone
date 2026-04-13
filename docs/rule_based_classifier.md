# Rule-Based Intent Classifier

## Overview

The rule-based classifier (`src/ai_corpus/intent_classifier.py`) assigns each AI-related text chunk two labels: an **intent level** (how far along a bank is in adopting AI) and one or more **application categories** (what type of AI they're talking about).

## Step 1: Filtering

Before classification, chunks are filtered through `has_ai_anchor()` (`src/ai_corpus/pipeline.py`). Only chunks containing at least one AI keyword pass through:

- "artificial intelligence", "machine learning", "generative ai", "genai", "large language model", "llm", "chatbot", "copilot", "ai"

This reduced 256,385 total chunks to 2,757 AI-relevant chunks.

## Step 2: Intent Classification

Each chunk's text is scanned for signal words, checked from highest level down (4 → 1). The first match wins.

| Level | Label | Signal Words |
|-------|-------|-------------|
| 4 | Scaling | expanding, enterprise-wide, firm-wide, increasing, doubling down, across all, scaled, scaling |
| 3 | Deploying | launched, implemented, using, rolled out, live, operational, deployed, in production |
| 2 | Committing | investing, budgeting, building, planning, allocating, developing, committed, funding |
| 1 | Exploring | investigating, considering, evaluating, studying, piloting, researching, exploring, assessing |

If no signal words match, the chunk defaults to level 1 (Exploring).

## Step 3: Application Category Tagging

Each chunk is checked against keyword lists for all 6 categories. A chunk can match multiple categories.

| Category | Keywords |
|----------|----------|
| GenAI / LLMs | chatbot, copilot, generative ai, large language model, llm, gpt, prompt, genai |
| Predictive ML | credit scoring, forecasting, propensity model, regression, prediction, predictive |
| NLP / Text | document processing, sentiment analysis, entity extraction, text mining, natural language, nlp |
| Computer Vision | check imaging, id verification, facial recognition, ocr, computer vision, image |
| RPA / Automation | process automation, straight-through processing, workflow automation, robotic, rpa, automation |
| Fraud / Risk Models | fraud detection, aml, transaction monitoring, anomaly detection, risk model, anti-money |

## Output

Each classified chunk produces a JSON record:

```json
{
  "chunk_id": "BAC_transcript_2024_Q4__chunk_0012",
  "ticker": "BAC",
  "bank_name": "Bank of America",
  "source_type": "transcript",
  "period_year": 2024,
  "period_quarter": 4,
  "intent_level": 3,
  "intent_label": "Deploying",
  "app_categories": ["GenAI / LLMs", "Fraud / Risk Models"],
  "confidence": 0.5,
  "evidence_snippet": "We have deployed generative AI tools across our fraud..."
}
```

All rule-based results have `confidence: 0.5`. LLM-classified results (when Qwen is available) have variable confidence scores.

## Results Summary

- 256,385 chunks processed → 2,757 classified
- Intent distribution: 36% Exploring, 9% Committing, 34% Deploying, 20% Scaling
- Top application area: GenAI / LLMs (557 mentions, 49% of tagged categories)
