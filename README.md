# BUFN403 Capstone

This repo now contains two parallel workflows:

- `scripts/classify_bank_ai.py` for the earlier transcript and 10-K scoring pipeline.
- `scripts/ai_corpus_pipeline.py` for the broader AI-usage corpus audit, normalization, indexing, retrieval, and topic-analysis workflow.

## End Goal

Build a source-grounded research corpus that answers what each of the 50 banks says about using AI across:

- investor transcripts
- SEC filings
- call reports
- MRA/MRIA or other supervisory documents

The new pipeline keeps Gemini out of the critical path for now and uses Qwen-first answer generation with a local hybrid store:

- Chroma for narrative text chunks
- DuckDB plus Parquet for structured call-report data

## Expected Inputs

Current repo inputs:

- `AI_Bank_Classification.csv` as the 50-bank roster
- `10K_10Q_8K_DEF14A_combined_data.zip`
- `transcripts_final-20260304T030232Z-1-001.zip`

Manual drop zones for anything you find outside those archives:

- `manual_sources/call_reports/`
- `manual_sources/mra_mria/`
- `manual_sources/transcripts/`
- `manual_sources/sec/`

Manual files should ideally include the ticker and period in the filename, for example:

- `BAC_call_report_2024_Q1.json`
- `WFC_mria_2025_Q2.pdf`
- `JPM_10-K_2025_Q4.html`

## Commands

Build the completeness manifest:

```bash
python scripts/ai_corpus_pipeline.py build-manifest
```

Attempt to fetch missing public documents:

```bash
python scripts/ai_corpus_pipeline.py acquire-missing
```

Normalize available files and emit `chunks.jsonl` plus structured tables:

```bash
python scripts/ai_corpus_pipeline.py normalize-corpus
```

Build the Chroma and DuckDB artifacts:

```bash
python scripts/ai_corpus_pipeline.py build-index
```

Search the corpus:

```bash
python scripts/ai_corpus_pipeline.py search --question "What does BAC say about AI strategy?"
```

Ask the corpus:

```bash
python scripts/ai_corpus_pipeline.py ask --question "What do BAC and WFC say about AI governance?"
```

Optimize the prompt template over the benchmark set:

```bash
python scripts/ai_corpus_pipeline.py optimize-prompts
```

Build topic findings and plots:

```bash
python scripts/ai_corpus_pipeline.py build-topic-findings
```

Classify AI chunks with Qwen:

```bash
python scripts/ai_corpus_pipeline.py classify
```

Run the same classification step on a Google Colab GPU and allow model download:

```bash
python scripts/ai_corpus_pipeline.py classify --prefer-local --allow-model-download
```

Resume a partially completed classification run and emit paced progress logs:

```bash
python -u scripts/ai_corpus_pipeline.py classify --prefer-local --allow-model-download --resume --log-every 5
```

If a Colab run is interrupted, rerunning that same command will:

- repair a malformed trailing `classifications.jsonl` record if the last append was cut off
- skip already completed `chunk_id`s
- exit early without reloading Qwen if everything is already classified
- update `artifacts/ai_corpus/classification_progress.json` with current-session rate and ETA fields

Build the dashboard-ready score artifacts from `classifications.jsonl`:

```bash
python scripts/ai_corpus_pipeline.py score
```

Generate a per-bank AI summary report:

```bash
python scripts/ai_corpus_pipeline.py build-bank-summaries
```

## Google Colab

If you want to run Qwen on a Colab GPU, the simplest path is:

1. In Colab, switch the runtime to a GPU:
   `Runtime -> Change runtime type -> T4 GPU` (or better if available)
2. Get the repo into the notebook:
   - If your branch is pushed to GitHub:
     ```bash
     !git clone https://github.com/Ostailor/BUFN403_Capstone.git
     %cd BUFN403_Capstone
     ```
   - If your branch is only local:
     zip the repo on your machine, upload it to Colab or Google Drive, then unzip:
     ```bash
     from google.colab import files
     uploaded = files.upload()
     ```
3. Install dependencies:
   ```bash
   !pip install -r requirements.txt
   ```
4. Put your input files into the repo root or mount Drive and copy them into place:
   - `AI_Bank_Classification.csv`
   - `10K_10Q_8K_DEF14A_combined_data.zip`
   - `transcripts_final-20260304T030232Z-1-001.zip`
5. Run the pipeline:
   ```bash
   !python scripts/ai_corpus_pipeline.py normalize-corpus
   !python -u scripts/ai_corpus_pipeline.py classify --prefer-local --allow-model-download --resume --log-every 5
   !python scripts/ai_corpus_pipeline.py score
   ```
6. Download the outputs you need from `artifacts/ai_corpus/`, especially:
   - `classifications.jsonl`
   - `bank_composite_scores.csv`
   - `quarterly_progression.csv`
   - `app_category_matrix.csv`

## Outputs

The new pipeline writes under `artifacts/ai_corpus/`:

- `document_manifest.csv`
- `acquisition_log.csv`
- `normalized/documents/*.json`
- `normalized/tables/call_reports.parquet`
- `chunks.jsonl`
- `corpus.duckdb`
- `index/` for the Chroma collection
- `compiled_prompt.json`
- `classifications.jsonl`
- `bank_composite_scores.csv`
- `quarterly_progression.csv`
- `app_category_matrix.csv`
- `bank_ai_summaries.csv`
- `bank_ai_summaries.json`
- `bank_ai_summaries.md`
- `topic_findings.csv`
- `plots/*.png`

## Notes

- SEC acquisition uses official SEC JSON submissions endpoints when available.
- Call-report acquisition uses the official FDIC BankFind financial API.
- MRA/MRIA acquisition is logged, but many of those documents are not public; manual collection is expected.
- `classify` is now fail-fast: if Qwen generation fails for a chunk, the run raises instead of silently falling back to regex heuristics.
- `classify` now writes progress incrementally and resumes from existing `classifications.jsonl` records by default, so reruns continue from the next unfinished chunk.
- Progress metadata is written to `artifacts/ai_corpus/classification_progress.json`.
- Resume-safe progress metadata includes session-scoped throughput and ETA so Colab logs reflect the current run rather than historical completed chunks.
- For Colab GPU sessions, `--allow-model-download` lets the local Qwen path pull weights instead of requiring a pre-populated local cache.
- The dashboard now expects canonical score artifacts or `classifications.jsonl`; it no longer invents synthetic ranking data when those files are missing.
