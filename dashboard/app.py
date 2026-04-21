from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st

st.set_page_config(page_title="BUFN403 Capstone Dashboard", layout="wide")

from dashboard.core.registry import build_navigation, discover_teams

teams = discover_teams()
nav = build_navigation(teams)

if not nav:
    st.title("BUFN403 Capstone Dashboard")
    st.info(
        "No teams registered yet. Add a team under `dashboard/teams/<slug>/` with a "
        "`manifest.toml` and pages. See AGENTS.md at the repo root."
    )
else:
    st.navigation(nav).run()
