#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.ai_corpus.config import CorpusPaths
from src.ai_corpus.pipeline import (
    acquire_missing,
    ask,
    build_bank_ai_summaries,
    build_index,
    build_manifest,
    build_topic_findings,
    normalize_corpus,
    optimize_prompts,
    search,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AI usage corpus audit and Qwen-first RAG pipeline.")
    parser.add_argument("--output-dir", type=Path, default=CorpusPaths().output_dir, help="Directory for generated artifacts")
    parser.add_argument("--manual-source-dir", type=Path, default=CorpusPaths().manual_source_dir, help="Directory for manually collected documents")
    parser.add_argument("--roster-csv", type=Path, default=CorpusPaths().roster_csv, help="CSV defining the 50-bank roster")
    parser.add_argument("--sec-zip", type=Path, default=CorpusPaths().sec_zip, help="SEC filings zip archive")
    parser.add_argument("--transcript-zip", type=Path, default=CorpusPaths().transcript_zip, help="Transcripts zip archive")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("build-manifest", help="Build document completeness manifest")
    subparsers.add_parser("acquire-missing", help="Attempt to fetch missing public documents")
    subparsers.add_parser("normalize-corpus", help="Normalize available documents and emit chunks")

    index_parser = subparsers.add_parser("build-index", help="Build Chroma and DuckDB artifacts")
    index_parser.add_argument("--embedding-model", default=None, help="Embedding model name or 'hash' for tests")

    search_parser = subparsers.add_parser("search", help="Search the indexed corpus")
    search_parser.add_argument("--question", required=True)
    search_parser.add_argument("--ticker", default="")
    search_parser.add_argument("--source-type", default="")
    search_parser.add_argument("--form-type", default="")
    search_parser.add_argument("--embedding-model", default=None)

    ask_parser = subparsers.add_parser("ask", help="Ask a question against the indexed corpus")
    ask_parser.add_argument("--question", required=True)
    ask_parser.add_argument("--ticker", default="")
    ask_parser.add_argument("--source-type", default="")
    ask_parser.add_argument("--form-type", default="")
    ask_parser.add_argument("--embedding-model", default=None)

    subparsers.add_parser("optimize-prompts", help="Optimize prompt template over benchmark questions")
    subparsers.add_parser("build-topic-findings", help="Generate topic findings and plots")
    summary_parser = subparsers.add_parser("build-bank-summaries", help="Generate per-bank AI summaries")
    summary_parser.add_argument("--embedding-model", default=None)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    paths = CorpusPaths(
        output_dir=args.output_dir,
        manual_source_dir=args.manual_source_dir,
        roster_csv=args.roster_csv,
        sec_zip=args.sec_zip,
        transcript_zip=args.transcript_zip,
    )

    if args.command == "build-manifest":
        payload = [row.as_dict() for row in build_manifest(paths=paths)]
    elif args.command == "acquire-missing":
        payload = [row.as_dict() for row in acquire_missing(paths=paths)]
    elif args.command == "normalize-corpus":
        payload = normalize_corpus(paths=paths)
    elif args.command == "build-index":
        payload = build_index(paths=paths, embedding_model=args.embedding_model)
    elif args.command == "search":
        filters = {
            key: value
            for key, value in {
                "ticker": args.ticker,
                "source_type": args.source_type,
                "form_type": args.form_type,
            }.items()
            if value
        }
        payload = search(
            args.question,
            paths=paths,
            filters=filters or None,
            embedding_model=args.embedding_model,
        )
    elif args.command == "ask":
        filters = {
            key: value
            for key, value in {
                "ticker": args.ticker,
                "source_type": args.source_type,
                "form_type": args.form_type,
            }.items()
            if value
        }
        payload = ask(
            args.question,
            paths=paths,
            filters=filters or None,
            embedding_model=args.embedding_model,
        ).as_dict()
    elif args.command == "optimize-prompts":
        payload = optimize_prompts(paths=paths)
    elif args.command == "build-topic-findings":
        payload = build_topic_findings(paths=paths)
    elif args.command == "build-bank-summaries":
        payload = build_bank_ai_summaries(paths=paths, embedding_model=args.embedding_model)
    else:
        raise SystemExit(f"Unsupported command: {args.command}")

    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
