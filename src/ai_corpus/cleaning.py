from __future__ import annotations

import html
import re
from pathlib import Path

import pdfplumber
from pypdf import PdfReader


def normalize_whitespace(text: str) -> str:
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_transcript_text(text: str) -> str:
    lines = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        lower = line.lower()
        if lower.startswith("image source:") or lower.startswith("need a quote"):
            continue
        lines.append(line)
    return normalize_whitespace("\n".join(lines))


def clean_sec_html(text: str) -> str:
    cleaned = text
    cleaned = re.sub(r"(?is)<(script|style)\b.*?>.*?</\1>", " ", cleaned)
    cleaned = re.sub(r"(?is)<[^>]*display\s*:\s*none[^>]*>.*?</[^>]+>", " ", cleaned)
    cleaned = re.sub(r"(?is)<ix:[^>]*>.*?</ix:[^>]*>", " ", cleaned)
    cleaned = re.sub(r"(?is)</?ix:[^>]*>", " ", cleaned)
    cleaned = re.sub(
        r"(?is)</?(xbrli|xbrldi|dei|us-gaap|link|xbrl|ixt|ixt-sec|measure|context|unit)[^>]*>",
        " ",
        cleaned,
    )
    cleaned = re.sub(r"(?is)</?(div|p|br|tr|li|h1|h2|h3|h4|h5|h6|table|section)[^>]*>", "\n", cleaned)
    cleaned = re.sub(r"(?is)<[^>]+>", " ", cleaned)
    cleaned = html.unescape(cleaned)
    cleaned = re.sub(r"\n\s+\n", "\n\n", cleaned)
    return normalize_whitespace(cleaned)


def clean_plain_text(text: str) -> str:
    return normalize_whitespace(text)


def extract_text_from_pdf(path: Path) -> str:
    pages: list[str] = []
    try:
        reader = PdfReader(str(path))
        for page in reader.pages:
            pages.append(page.extract_text() or "")
    except Exception:
        pages = []
    if not "".join(pages).strip():
        with pdfplumber.open(path) as pdf:
            pages = [(page.extract_text() or "") for page in pdf.pages]
    return normalize_whitespace("\n\n".join(pages))


def extract_text_from_file(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md", ".csv"}:
        return clean_plain_text(path.read_text(encoding="utf-8", errors="ignore"))
    if suffix in {".html", ".htm"}:
        return clean_sec_html(path.read_text(encoding="utf-8", errors="ignore"))
    if suffix == ".pdf":
        return extract_text_from_pdf(path)
    return clean_plain_text(path.read_text(encoding="utf-8", errors="ignore"))


def split_sections(text: str, source_type: str) -> list[tuple[str, str]]:
    if not text.strip():
        return []

    if source_type == "transcript":
        sections: list[tuple[str, str]] = []
        current_title = "Transcript"
        current_lines: list[str] = []
        for line in text.splitlines():
            if re.match(r"^[A-Z][A-Za-z .,'&/-]{1,80}:", line):
                if current_lines:
                    sections.append((current_title, normalize_whitespace("\n".join(current_lines))))
                    current_lines = []
                current_title = line.split(":", 1)[0].strip()
            current_lines.append(line)
        if current_lines:
            sections.append((current_title, normalize_whitespace("\n".join(current_lines))))
        return sections or [("Transcript", text)]

    sections = []
    current_title = "Document"
    current_lines = []
    for paragraph in re.split(r"\n{2,}", text):
        para = paragraph.strip()
        if not para:
            continue
        if len(para) < 120 and (para.isupper() or para.endswith(":")):
            if current_lines:
                sections.append((current_title, normalize_whitespace("\n\n".join(current_lines))))
                current_lines = []
            current_title = para.title()
            continue
        current_lines.append(para)
    if current_lines:
        sections.append((current_title, normalize_whitespace("\n\n".join(current_lines))))
    return sections or [("Document", text)]
