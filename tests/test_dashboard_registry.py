from __future__ import annotations

import sys
import threading
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# The registry module imports streamlit lazily inside `build_navigation`,
# but `TeamManifest`/`discover_teams` do not. The dashboard.teams package
# contains `data_loader.py` modules that import streamlit at top level —
# importing `dashboard.core.registry` is fine without streamlit, but we
# still require it because the fixture below needs a Streamlit context
# to construct `st.Page` objects. Gate the whole module on streamlit.
pytest.importorskip("streamlit")

from dashboard.core.paths import project_root, team_artifacts_dir  # noqa: E402
from dashboard.core.registry import (  # noqa: E402
    TeamManifest,
    build_navigation,
    discover_teams,
)


EXPECTED_SLUGS = {
    "ai_classification_intent",
    "ai_risk",
    "crypto_classification_intent",
    "crypto_risk",
    "private_credit_classification_intent",
    "private_credit_risk",
}

EXPECTED_MATRIX = {
    ("AI", "Classification & Intent"): "ai_classification_intent",
    ("AI", "Risk Analysis"): "ai_risk",
    ("Crypto", "Classification & Intent"): "crypto_classification_intent",
    ("Crypto", "Risk Analysis"): "crypto_risk",
    ("Private Credit", "Classification & Intent"): "private_credit_classification_intent",
    ("Private Credit", "Risk Analysis"): "private_credit_risk",
}


@pytest.fixture
def streamlit_ctx():
    """Attach a minimal Streamlit ScriptRunContext to the current thread.

    `st.Page` short-circuits (leaving `_title` unset) when no context is
    bound, which breaks attribute access during navigation tests. This
    fixture creates a throw-away context and detaches it on teardown.
    """
    from streamlit.runtime.scriptrunner_utils.script_run_context import (
        ScriptRunContext,
        add_script_run_ctx,
        get_script_run_ctx,
    )
    from streamlit.runtime.pages_manager import PagesManager
    from streamlit.runtime.state import SafeSessionState, SessionState

    thread = threading.current_thread()
    prev_ctx = get_script_run_ctx()

    main_script = str(project_root() / "dashboard" / "app.py")
    ctx = ScriptRunContext(
        session_id="test-session",
        _enqueue=lambda msg: None,
        query_string="",
        session_state=SafeSessionState(SessionState(), lambda: None),
        uploaded_file_mgr=None,
        main_script_path=main_script,
        user_info={},
        fragment_storage=None,
        pages_manager=PagesManager(main_script),
    )
    add_script_run_ctx(thread, ctx)
    try:
        yield
    finally:
        add_script_run_ctx(thread, prev_ctx)


def test_discover_teams_finds_all_six() -> None:
    teams = discover_teams()
    assert len(teams) == 6
    assert {t.team_slug for t in teams} == EXPECTED_SLUGS


def test_discover_teams_section_subteam_assignment() -> None:
    teams = discover_teams()
    valid_sections = {"AI", "Crypto", "Private Credit"}
    valid_subteams = {"Classification & Intent", "Risk Analysis"}

    by_key: dict[tuple[str, str], str] = {}
    for team in teams:
        assert isinstance(team, TeamManifest)
        assert team.section in valid_sections
        assert team.subteam in valid_subteams
        by_key[(team.section, team.subteam)] = team.team_slug

    assert by_key == EXPECTED_MATRIX


def test_build_navigation_groups_by_section(streamlit_ctx) -> None:
    teams = discover_teams()
    nav = build_navigation(teams)

    assert list(nav.keys()) == ["AI", "Crypto", "Private Credit"]
    for section, pages in nav.items():
        assert len(pages) >= 1, f"section {section!r} has no pages"

    ai_titles = {page.title for page in nav["AI"]}
    expected_titles = {
        "Classification & Intent · Leaderboard",
        "Classification & Intent · Bank Deep Dive",
        "Classification & Intent · Compare Banks",
        "Classification & Intent · Market Overview",
    }
    assert expected_titles <= ai_titles


def test_manifest_artifacts_dir_resolves() -> None:
    team_page = (
        project_root()
        / "dashboard"
        / "teams"
        / "ai_classification_intent"
        / "pages"
        / "leaderboard.py"
    )
    resolved = team_artifacts_dir(team_page)
    assert resolved is not None
    assert resolved == (project_root() / "artifacts" / "ai_corpus").resolve()
    assert str(resolved).endswith(str(Path("artifacts") / "ai_corpus"))


def test_stub_team_has_empty_artifacts_dir() -> None:
    stub_page = (
        project_root()
        / "dashboard"
        / "teams"
        / "crypto_risk"
        / "pages"
        / "placeholder.py"
    )
    assert team_artifacts_dir(stub_page) is None


def test_invalid_section_is_skipped(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bogus_dir = project_root() / "dashboard" / "teams" / "_bogus_test"
    bogus_dir.mkdir(parents=False, exist_ok=False)
    try:
        (bogus_dir / "manifest.toml").write_text(
            """
[team]
section       = "Bogus"
subteam       = "Classification & Intent"
display_name  = "Bogus"
owners        = []
order         = 99

[data]
artifacts_dir = ""

[[pages]]
path  = "pages/placeholder.py"
title = "Placeholder"
order = 1
""".lstrip(),
            encoding="utf-8",
        )

        teams = discover_teams()
        slugs = {t.team_slug for t in teams}
        assert "_bogus_test" not in slugs
        # The real six are still there; discovery didn't raise.
        assert EXPECTED_SLUGS <= slugs
    finally:
        # Teardown: remove bogus manifest + dir.
        manifest = bogus_dir / "manifest.toml"
        if manifest.exists():
            manifest.unlink()
        if bogus_dir.exists():
            bogus_dir.rmdir()
