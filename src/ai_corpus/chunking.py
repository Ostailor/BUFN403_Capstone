from __future__ import annotations

from .models import ChunkRecord, NormalizedDocument
from .themes import tag_themes
from .utils import normalize_key, sha256_text


def approximate_token_count(text: str) -> int:
    return max(1, len(text.split()))


def split_into_chunks(text: str, *, max_tokens: int = 220, overlap_tokens: int = 40) -> list[str]:
    words = text.split()
    if len(words) <= max_tokens:
        return [text]
    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = min(len(words), start + max_tokens)
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start = max(0, end - overlap_tokens)
    return chunks


def build_chunks(document: NormalizedDocument) -> list[ChunkRecord]:
    section_chunks = split_into_chunks(document.cleaned_text)
    records: list[ChunkRecord] = []
    for index, chunk_text in enumerate(section_chunks):
        theme_tags = sorted(set(document.theme_tags + tag_themes(chunk_text)))
        chunk_id = f"{document.doc_id}__chunk_{index:04d}"
        quality_flags = []
        if len(chunk_text) < 100:
            quality_flags.append("short")
        if approximate_token_count(chunk_text) > 260:
            quality_flags.append("oversized")
        records.append(
            ChunkRecord(
                chunk_id=chunk_id,
                doc_id=document.doc_id,
                ticker=document.ticker,
                bank_name=document.bank_name,
                source_type=document.source_type,
                form_type=document.form_type,
                period_year=document.period_year,
                period_quarter=document.period_quarter,
                filing_or_issue_date=document.filing_or_issue_date,
                section_title=document.section_title,
                chunk_index=index,
                chunk_text=chunk_text,
                source_path_or_url=document.source_path_or_url,
                content_hash=sha256_text(f"{document.doc_id}:{chunk_text}"),
                quality_flags=",".join(quality_flags),
                theme_tags=",".join(theme_tags),
            )
        )
    return records
