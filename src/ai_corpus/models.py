from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class BankRecord:
    ticker: str
    bank_name: str


@dataclass(slots=True)
class DiscoveredDocument:
    doc_id: str
    ticker: str
    bank_name: str
    source_type: str
    form_type: str
    period_year: int | None
    period_quarter: int | None
    filing_or_issue_date: str
    storage_kind: str
    source_path_or_url: str
    local_path: str
    content_type: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ManifestRow:
    ticker: str
    bank_name: str
    source_type: str
    form_type: str
    period_year: int | None
    period_quarter: int | None
    period_label: str
    expected_doc_count: int | None
    observed_doc_count: int
    status: str
    status_detail: str
    storage_kind: str
    local_refs: str
    source_urls: str
    manual_search_hint: str
    notes: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class AcquisitionLogRow:
    timestamp_utc: str
    ticker: str
    bank_name: str
    source_type: str
    form_type: str
    period_label: str
    attempt_type: str
    query_or_url: str
    outcome: str
    saved_path: str
    notes: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class NormalizedDocument:
    doc_id: str
    ticker: str
    bank_name: str
    source_type: str
    form_type: str
    period_year: int | None
    period_quarter: int | None
    filing_or_issue_date: str
    section_title: str
    source_path_or_url: str
    storage_kind: str
    cleaned_text: str
    tables: list[dict[str, Any]] = field(default_factory=list)
    theme_tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def output_path(self, documents_dir: Path) -> Path:
        return documents_dir / f"{self.doc_id}.json"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ChunkRecord:
    chunk_id: str
    doc_id: str
    ticker: str
    bank_name: str
    source_type: str
    form_type: str
    period_year: int | None
    period_quarter: int | None
    filing_or_issue_date: str
    section_title: str
    chunk_index: int
    chunk_text: str
    source_path_or_url: str
    content_hash: str
    quality_flags: str
    theme_tags: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class SearchHit:
    rank: int
    chunk_id: str
    score: float
    ticker: str
    bank_name: str
    source_type: str
    form_type: str
    period_year: int | None
    period_quarter: int | None
    section_title: str
    source_path_or_url: str
    theme_tags: list[str]
    chunk_text: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class StructuredRef:
    ticker: str
    report_date: str
    cert: int | None
    metric_name: str
    metric_value: float | str
    source_url: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class AskResult:
    answer_text: str
    citations: list[str]
    retrieved_chunk_ids: list[str]
    structured_data_refs: list[dict[str, Any]]
    theme_tags: list[str]
    confidence: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)
