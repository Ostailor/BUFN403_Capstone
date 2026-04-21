from __future__ import annotations

import streamlit as st


def missing_data_warning(message: str) -> None:
    st.warning(message)
    st.stop()


def placeholder_page(section: str, subteam: str) -> None:
    st.title(f"{section} · {subteam}")
    st.info(
        "This team hasn't added pages yet. See AGENTS.md at the repo root, or run "
        "`python scripts/new_team_page.py --team <slug> --page-slug <name> --title '<Title>'`."
    )


def section_header(title: str, subtitle: str | None = None) -> None:
    st.header(title)
    if subtitle:
        st.caption(subtitle)
