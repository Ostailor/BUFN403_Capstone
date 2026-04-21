from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dashboard.core.paths import project_root

VALID_SECTIONS = ("AI", "Crypto", "Private Credit")
VALID_SUBTEAMS = ("Classification & Intent", "Risk Analysis")
SECTION_ORDER = {name: i for i, name in enumerate(VALID_SECTIONS)}


def _warn(msg: str) -> None:
    try:
        import streamlit as st  # type: ignore

        st.warning(msg)
    except Exception:
        print(msg)


@dataclass
class TeamManifest:
    section: str
    subteam: str
    display_name: str
    owners: list[str]
    order: int
    artifacts_dir: str
    pages: list[dict[str, Any]]
    team_slug: str
    team_dir: Path


def _parse_manifest(manifest_path: Path) -> TeamManifest | None:
    try:
        data = tomllib.loads(manifest_path.read_text())
    except Exception as exc:
        _warn(f"Skipping {manifest_path}: failed to parse ({exc})")
        return None

    team = data.get("team") or {}
    section = team.get("section")
    subteam = team.get("subteam")
    display_name = team.get("display_name")
    owners = team.get("owners") or []
    order = team.get("order", 0)

    if section not in VALID_SECTIONS:
        _warn(f"Skipping {manifest_path}: invalid section {section!r}")
        return None
    if subteam not in VALID_SUBTEAMS:
        _warn(f"Skipping {manifest_path}: invalid subteam {subteam!r}")
        return None
    if not display_name:
        _warn(f"Skipping {manifest_path}: missing display_name")
        return None

    artifacts_dir = (data.get("data") or {}).get("artifacts_dir", "") or ""

    pages_raw = data.get("pages") or []
    pages: list[dict[str, Any]] = []
    for p in pages_raw:
        path = p.get("path")
        title = p.get("title")
        if not path or not title:
            _warn(f"Skipping {manifest_path}: page missing path/title")
            return None
        pages.append(
            {
                "path": path,
                "title": title,
                "icon": p.get("icon"),
                "order": p.get("order", 0),
            }
        )

    tdir = manifest_path.parent
    return TeamManifest(
        section=section,
        subteam=subteam,
        display_name=display_name,
        owners=list(owners),
        order=int(order),
        artifacts_dir=artifacts_dir,
        pages=pages,
        team_slug=tdir.name,
        team_dir=tdir,
    )


def discover_teams() -> list[TeamManifest]:
    teams_root = project_root() / "dashboard" / "teams"
    if not teams_root.exists():
        return []

    found: list[TeamManifest] = []
    for child in sorted(teams_root.iterdir()):
        if not child.is_dir():
            continue
        manifest = child / "manifest.toml"
        if not manifest.exists():
            continue
        parsed = _parse_manifest(manifest)
        if parsed is not None:
            found.append(parsed)

    found.sort(key=lambda t: (SECTION_ORDER.get(t.section, 99), t.order, t.team_slug))
    return found


def build_navigation(teams: list[TeamManifest]):
    import streamlit as st  # type: ignore

    nav: dict[str, list] = {section: [] for section in VALID_SECTIONS}

    for section in VALID_SECTIONS:
        section_teams = [t for t in teams if t.section == section]
        entries: list[tuple[int, int, Any]] = []
        for team in section_teams:
            for page in team.pages:
                page_path = team.team_dir / page["path"]
                title = f"{team.subteam} · {page['title']}"
                url_path = f"{team.team_slug}_{Path(page['path']).stem}"
                kwargs: dict[str, Any] = {"title": title, "url_path": url_path}
                if page.get("icon"):
                    kwargs["icon"] = page["icon"]
                entries.append((team.order, page.get("order", 0), st.Page(str(page_path), **kwargs)))
        entries.sort(key=lambda e: (e[0], e[1]))
        nav[section] = [e[2] for e in entries]

    return {section: pages for section, pages in nav.items() if pages}
