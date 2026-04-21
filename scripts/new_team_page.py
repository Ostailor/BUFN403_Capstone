"""Scaffold a new page inside an existing dashboard team.

Usage:
    python scripts/new_team_page.py --team crypto_risk \\
        --page-slug var_dashboard --title "VaR Dashboard" \\
        [--icon ":material/chart:"]

Stdlib only. Does not import Streamlit or pandas.
"""
from __future__ import annotations

import argparse
import re
import sys
import tomllib
from pathlib import Path


SLUG_RE = re.compile(r"^[a-z][a-z0-9_]*$")

PAGE_TEMPLATE = '''import streamlit as st
import pandas as pd

from dashboard.core.ui import missing_data_warning
# from dashboard.teams.{slug}.data_loader import load_something

st.title("{title}")

# data = load_something()
# if data.empty:
#     missing_data_warning("No data for {slug}. Drop your artifact into artifacts/{slug}/.")

st.write("This is a new page. Replace this stub with your analysis.")
'''


def project_root() -> Path:
    here = Path(__file__).resolve()
    for parent in [here, *here.parents]:
        if (parent / "dashboard").is_dir() and (parent / "artifacts").exists():
            return parent
    return here.parents[1]


def list_team_slugs(teams_dir: Path) -> list[str]:
    if not teams_dir.is_dir():
        return []
    slugs = []
    for child in sorted(teams_dir.iterdir()):
        if child.is_dir() and (child / "manifest.toml").exists():
            slugs.append(child.name)
    return slugs


def next_order(manifest_data: dict) -> int:
    pages = manifest_data.get("pages") or []
    if not pages:
        return 1
    orders = [int(p.get("order", 0)) for p in pages if isinstance(p, dict)]
    return (max(orders) if orders else 0) + 1


def render_page_block(page_slug: str, title: str, icon: str | None, order: int) -> str:
    lines = [
        "",
        "[[pages]]",
        f'path  = "pages/{page_slug}.py"',
        f'title = "{title}"',
    ]
    if icon:
        lines.append(f'icon  = "{icon}"')
    lines.append(f"order = {order}")
    lines.append("")
    return "\n".join(lines)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add a page to an existing dashboard team.",
    )
    parser.add_argument("--team", required=True, help="Team slug, e.g. crypto_risk.")
    parser.add_argument(
        "--page-slug",
        required=True,
        help="Filename stem for the new page (snake_case, starts with a letter).",
    )
    parser.add_argument("--title", required=True, help="Display title for the page.")
    parser.add_argument(
        "--icon",
        default=None,
        help='Optional Streamlit icon, e.g. ":material/chart:".',
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    root = project_root()
    teams_dir = root / "dashboard" / "teams"
    valid_slugs = list_team_slugs(teams_dir)

    if args.team not in valid_slugs:
        print(f"error: unknown team slug {args.team!r}.", file=sys.stderr)
        print("valid team slugs:", file=sys.stderr)
        for slug in valid_slugs:
            print(f"  - {slug}", file=sys.stderr)
        return 2

    if not SLUG_RE.match(args.page_slug):
        print(
            f"error: --page-slug {args.page_slug!r} must match ^[a-z][a-z0-9_]*$.",
            file=sys.stderr,
        )
        return 2

    team_dir = teams_dir / args.team
    manifest_path = team_dir / "manifest.toml"
    pages_dir = team_dir / "pages"
    page_path = pages_dir / f"{args.page_slug}.py"

    if page_path.exists():
        print(f"error: {page_path} already exists; refusing to overwrite.", file=sys.stderr)
        return 1

    manifest_data = tomllib.loads(manifest_path.read_text())
    order = next_order(manifest_data)

    pages_dir.mkdir(parents=True, exist_ok=True)
    page_path.write_text(PAGE_TEMPLATE.format(slug=args.team, title=args.title))

    block = render_page_block(args.page_slug, args.title, args.icon, order)
    existing = manifest_path.read_text()
    if not existing.endswith("\n"):
        existing += "\n"
    manifest_path.write_text(existing + block)

    rel_page = page_path.relative_to(root)
    print(f"Created {rel_page} (order={order}).")
    print("Reload the dashboard with: streamlit run dashboard/app.py")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
