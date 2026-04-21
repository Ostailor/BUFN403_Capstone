from __future__ import annotations

import functools
import tomllib
from pathlib import Path


@functools.lru_cache(maxsize=1)
def project_root() -> Path:
    here = Path(__file__).resolve()
    for parent in [here, *here.parents]:
        if (parent / "artifacts").exists() or (parent / "README.md").exists():
            return parent
    return here.parent


def team_dir(reference_file: str | Path) -> Path:
    p = Path(reference_file).resolve()
    for parent in p.parents:
        if parent.parent.name == "teams" and parent.parent.parent.name == "dashboard":
            return parent
    raise ValueError(f"{reference_file!s} is not inside dashboard/teams/<slug>/")


def team_artifacts_dir(reference_file: str | Path) -> Path | None:
    tdir = team_dir(reference_file)
    manifest = tdir / "manifest.toml"
    if not manifest.exists():
        return None
    data = tomllib.loads(manifest.read_text())
    rel = (data.get("data") or {}).get("artifacts_dir", "")
    if not rel:
        return None
    return (project_root() / rel).resolve()
