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

Generate a per-bank AI summary report:

```bash
python scripts/ai_corpus_pipeline.py build-bank-summaries
```

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
- `bank_ai_summaries.csv`
- `bank_ai_summaries.json`
- `bank_ai_summaries.md`
- `topic_findings.csv`
- `plots/*.png`

## Notes

- SEC acquisition uses official SEC JSON submissions endpoints when available.
- Call-report acquisition uses the official FDIC BankFind financial API.
- MRA/MRIA acquisition is logged, but many of those documents are not public; manual collection is expected.
- If your local environment cannot load a larger Qwen checkpoint, the answer layer falls back down the configured Qwen model list.
