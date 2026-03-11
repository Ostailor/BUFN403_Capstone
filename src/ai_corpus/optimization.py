from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .utils import ensure_dir, write_json

DEFAULT_PROMPT_TEMPLATES = {
    "baseline": (
        "Answer using only retrieved evidence. Quote or paraphrase cautiously and keep citations explicit."
    ),
    "citation_first": (
        "Answer only from retrieved evidence. Every claim must be anchored to chunk ids or structured refs."
    ),
    "abstain_first": (
        "If evidence is incomplete, answer 'insufficient evidence'. Prefer abstention over unsupported synthesis."
    ),
    "analyst_mode": (
        "Explain what the bank says about AI, name the use case or strategy, mention risk/governance if present, and cite every point."
    ),
}


@dataclass(slots=True)
class BenchmarkExample:
    question: str
    expected_terms: list[str]
    expected_theme_tags: list[str]
    should_abstain: bool = False


def load_benchmarks(benchmark_path: Path) -> list[BenchmarkExample]:
    if not benchmark_path.exists():
        return []
    examples = []
    for line in benchmark_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        examples.append(
            BenchmarkExample(
                question=payload["question"],
                expected_terms=list(payload.get("expected_terms", [])),
                expected_theme_tags=list(payload.get("expected_theme_tags", [])),
                should_abstain=bool(payload.get("should_abstain", False)),
            )
        )
    return examples


def optimize_prompt_templates(
    benchmark_path: Path,
    *,
    output_path: Path,
    evaluator: Callable[[str, BenchmarkExample], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    examples = load_benchmarks(benchmark_path)
    results = []
    for name, template in DEFAULT_PROMPT_TEMPLATES.items():
        score = 0.0
        example_results = []
        for example in examples:
            evaluation = evaluator(template, example) if evaluator else {
                "term_hits": sum(term.lower() in template.lower() for term in example.expected_terms),
                "theme_hits": sum(tag.replace("_", " ") in template.lower() for tag in example.expected_theme_tags),
                "abstain_support": 1 if ("insufficient evidence" in template.lower()) == example.should_abstain else 0,
            }
            score += (
                float(evaluation.get("term_hits", 0))
                + float(evaluation.get("theme_hits", 0))
                + float(evaluation.get("abstain_support", 0))
            )
            example_results.append(evaluation)
        results.append({"template_name": name, "template": template, "score": score, "details": example_results})
    results.sort(key=lambda row: row["score"], reverse=True)
    artifact = {
        "selected_template": results[0]["template"] if results else DEFAULT_PROMPT_TEMPLATES["citation_first"],
        "selected_template_name": results[0]["template_name"] if results else "citation_first",
        "results": results,
        "benchmark_count": len(examples),
    }
    write_json(artifact, output_path)
    return artifact
