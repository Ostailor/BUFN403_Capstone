from __future__ import annotations

import csv
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def normalize_key(text: str) -> str:
    lowered = text.lower()
    lowered = re.sub(r"[^a-z0-9]+", " ", lowered)
    return re.sub(r"\s+", " ", lowered).strip()


def slugify(text: str) -> str:
    lowered = normalize_key(text)
    return lowered.replace(" ", "_")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def join_values(values: Iterable[str]) -> str:
    return " | ".join(sorted({value for value in values if value}))


def write_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    ensure_dir(output_path.parent)
    if not rows:
        output_path.write_text("", encoding="utf-8")
        return
    with output_path.open("w", encoding="utf-8", newline="") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(data: Any, output_path: Path) -> None:
    ensure_dir(output_path.parent)
    output_path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def append_jsonl(rows: Iterable[dict[str, Any]], output_path: Path) -> None:
    ensure_dir(output_path.parent)
    with output_path.open("a", encoding="utf-8") as outfile:
        for row in rows:
            outfile.write(json.dumps(row, sort_keys=True) + "\n")


def period_label(year: int | None, quarter: int | None) -> str:
    if year is None:
        return "unknown"
    if quarter is None:
        return str(year)
    return f"{year}_Q{quarter}"


def quarter_end_date(year: int, quarter: int) -> str:
    month_day = {1: "0331", 2: "0630", 3: "0930", 4: "1231"}
    return f"{year}{month_day[quarter]}"


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def maybe_int(value: Any) -> int | None:
    if value in (None, "", "None"):
        return None
    return int(value)


def maybe_float(value: Any) -> float | None:
    if value in (None, "", "None"):
        return None
    return float(value)
