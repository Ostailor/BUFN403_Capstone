# AI · Classification & Intent

Owner: Aditya Dabeer

This team plugin surfaces the AI adoption posture of covered banks, using
Qwen-backed classifications of earnings-call chunks and other disclosures.

## Pages

- **Leaderboard** — ranks banks by Composite / Maturity / Breadth / Momentum
  with optional filtering by application category.
- **Bank Deep Dive** — per-bank intent distribution, application radar,
  quarterly maturity timeline, and an evidence table of classified chunks.
- **Compare Banks** — overlaid radar charts, side-by-side intent bars, and a
  leader-highlighted metrics table for 2–4 selected banks.
- **Market Overview** — industry heatmap, stacked intent distribution,
  maturity quadrant, and momentum movers across all banks.

## Data

Reads canonical CSV/JSONL artifacts from `artifacts/ai_corpus/`
(`bank_composite_scores.csv`, `quarterly_progression.csv`,
`app_category_matrix.csv`, `classifications.jsonl`). When a derived CSV is
missing, the loader rebuilds the view from `classifications.jsonl`.

See `docs/how_scoring_works.md` for the Maturity / Breadth / Momentum /
Composite scoring methodology.
