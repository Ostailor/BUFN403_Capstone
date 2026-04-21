# AGENTS.md

Read this file first. It tells a coding agent how to work inside the
BUFN403 Capstone dashboard without breaking other teams.

## 1. What this repo is

BUFN403 Capstone is an analysis of AI adoption across 50 U.S. banks
(earnings transcripts, SEC filings, FDIC call reports). What started as
a single team's workflow is now a shared Streamlit dashboard that hosts
six sibling teams from the class. Each team owns a folder under
`dashboard/teams/<slug>/` and ships its own pages, data loaders, and
artifacts. The app at `dashboard/app.py` discovers teams at runtime by
reading every team's `manifest.toml`.

## 2. Team matrix

The class is organized as a 3 x 2 matrix: three sections (AI, Crypto,
Private Credit) x two sub-teams per section (Classification & Intent,
Risk Analysis).

| Slug                               | Section         | Sub-team                 | Owners               | Data location                          |
|------------------------------------|-----------------|--------------------------|----------------------|----------------------------------------|
| `ai_classification_intent`         | AI              | Classification & Intent  | Aditya Dabeer (lead) | `artifacts/ai_corpus/`                 |
| `ai_risk`                          | AI              | Risk Analysis            | TBD                  | `artifacts/ai_risk/`                   |
| `crypto_classification_intent`     | Crypto          | Classification & Intent  | TBD                  | `artifacts/crypto_classification/`     |
| `crypto_risk`                      | Crypto          | Risk Analysis            | TBD                  | `artifacts/crypto_risk/`               |
| `private_credit_classification_intent` | Private Credit | Classification & Intent | TBD                 | `artifacts/private_credit_classification/` |
| `private_credit_risk`              | Private Credit  | Risk Analysis            | TBD                  | `artifacts/private_credit_risk/`       |

The `ai_classification_intent` team is the reference implementation;
the other five are stubs that need their own data and pages wired in.
The concrete data location for each team is whatever that team sets
in its `manifest.toml` under `[data].artifacts_dir`.

## 3. Repo layout you need to know

```
BUFN403_Capstone/
  AGENTS.md                          <- this file
  README.md
  requirements.txt
  docs/
    dashboard_teams.md               <- human-facing version of this file
    agent_handoff.md
    how_scoring_works.md             <- example of "mature team" docs
  dashboard/
    app.py                           <- entry point, loads every team
    core/
      paths.py                       <- project_root, team_dir, team_artifacts_dir
      registry.py                    <- manifest discovery + page loading
      ui.py                          <- missing_data_warning, placeholder_page
    teams/
      ai_classification_intent/
        manifest.toml
        data_loader.py
        pages/
          leaderboard.py
          bank_deep_dive.py
          compare_banks.py
          market_overview.py
      ai_risk/
      crypto_classification_intent/
      crypto_risk/
      private_credit_classification_intent/
      private_credit_risk/
  artifacts/
    ai_corpus/                       <- Aditya's AI team artifacts
    <team_slug>/                     <- each team drops files here
  scripts/
    new_team_page.py                 <- the scaffolder you will use
```

You should not need to edit anything under `dashboard/core/` or
`dashboard/app.py`. The framework is stable. Teams are plugins.

## 4. If you are onboarding a new contributor

Work through this decision tree. Do not skip steps.

### a. Identify the team

Ask the contributor: "Which section (AI / Crypto / Private Credit)
and which sub-team (Classification & Intent / Risk Analysis) are you
on?" Map the answer to one of the six slugs in the table above. The
folder `dashboard/teams/<slug>/` already exists. Do not create a new
team slug; the six are fixed.

### b. Locate the data

Ask: "What data do you have, and where does it live?" Common answers:
a CSV the team generated, a JSONL of labeled records, a Parquet dump,
or a directory of model outputs. Drop those files under
`artifacts/<slug>/` (create the directory if missing). Then open
`dashboard/teams/<slug>/manifest.toml` and set:

```toml
[data]
artifacts_dir = "artifacts/<slug>"
```

The path is relative to the repo root. `team_artifacts_dir(__file__)`
will resolve it at runtime.

### c. Decide on pages

Ask: "What questions should each page answer?" Each page answers one
question (e.g., "Which banks lead on intent?", "How does Bank X
compare to its peers?"). For every page the contributor wants, run
the scaffolder (see section 5). Keep page titles short and in title
case.

### d. Wire up the data loader

Open `dashboard/teams/<slug>/data_loader.py` and add a reader per
artifact the team will consume. Follow the pattern in section 6.
Cache every reader with `@st.cache_data`. Return an empty DataFrame
when data is missing; never raise.

### e. Verify locally

Run:

```
streamlit run dashboard/app.py
```

Confirm the team's section appears in the sidebar, each page loads
without a Python error, and pages with missing data display the
missing-data warning (not a stack trace).

## 5. How to add a page

Use the scaffolder. Do not hand-edit `manifest.toml` when a tool
exists. From the repo root:

```
python scripts/new_team_page.py \
  --team crypto_risk \
  --page-slug var_dashboard \
  --title "VaR Dashboard" \
  --icon ":material/chart:"
```

`--icon` is optional. The script:

1. Validates the team slug exists.
2. Reads the manifest and picks the next `order` value.
3. Creates `dashboard/teams/<slug>/pages/<page_slug>.py` from a
   template.
4. Appends a `[[pages]]` block to `manifest.toml`.

The generated page file looks like this:

```python
import streamlit as st
import pandas as pd

from dashboard.core.ui import missing_data_warning
# from dashboard.teams.<slug>.data_loader import load_<something>

st.title("<TITLE>")

# data = load_<something>()
# if data.empty:
#     missing_data_warning("No data for <team>. Drop your artifact into artifacts/<slug>/.")

st.write("This is a new page. Replace this stub with your analysis.")
```

Replace the commented imports and the stub body with real content.
Do not delete the `missing_data_warning` import unless the page has
no data dependency at all.

## 6. How to add a data loader

Every reader goes in `dashboard/teams/<slug>/data_loader.py`. Signature:

```python
from dashboard.core.paths import team_artifacts_dir
import streamlit as st
import pandas as pd

@st.cache_data
def load_my_data() -> pd.DataFrame:
    d = team_artifacts_dir(__file__)
    if d is None:
        return pd.DataFrame()
    p = d / "my_file.csv"
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p)
```

Rules:

- Always call `team_artifacts_dir(__file__)` to resolve the data
  directory. Do not hard-code paths.
- Always guard with an existence check and return an empty
  `pd.DataFrame` (or empty list / dict) when the file is missing.
- Always decorate with `@st.cache_data`. Return types must be
  picklable.
- One reader per artifact. Keep each function small and named for
  what it returns (`load_leaderboard`, `load_evidence`, not
  `load_data_1`).

## 7. Missing-data convention

The dashboard must never render fabricated or placeholder numbers.
If a loader returns an empty frame, the page must call:

```python
from dashboard.core.ui import missing_data_warning

missing_data_warning(
    "No data for <team_slug>. Drop your artifact into artifacts/<slug>/."
)
```

`missing_data_warning` displays a Streamlit warning and calls
`st.stop()`. This is the only acceptable empty-state behavior. Do
not catch it, do not fall through to mock data, do not synthesize
values from the page itself.

## 8. Local run & tests

From the repo root:

```
pip install -r requirements.txt
streamlit run dashboard/app.py
pytest tests/ -q
```

The dashboard starts on `http://localhost:8501`. Every team that has
a `manifest.toml` appears as a navigation section. Pages with no
artifacts still load and show the missing-data warning.

## 9. Conventions

- Page titles: title case ("Bank Deep Dive", not "bank deep dive").
- Axis labels: always include units ("Momentum (z-score)", "Assets
  (USD billions)").
- Colors: use Streamlit / Plotly defaults unless a chart needs a
  specific palette (e.g., diverging scales for z-scores). Do not
  hard-code brand colors.
- File names: `snake_case.py` for pages and loaders.
- Slugs: `^[a-z][a-z0-9_]*$`. The scaffolder enforces this.
- Keep imports explicit. No wildcard imports.
- No emoji in page output, titles, or commit messages.

## 10. Reference implementation

Point new contributors at:

- `dashboard/teams/ai_classification_intent/` — the canonical team
  layout. Use its `manifest.toml`, `data_loader.py`, and page files
  as templates.
- `docs/how_scoring_works.md` — the style of longer-form
  documentation a mature team produces once it has real results to
  explain.

When unsure what a stub team should look like once it is fleshed
out, mirror the AI Classification & Intent team.

## Pattern: Interactive parameter overrides

If your team has a scoring or aggregation formula with tunable weights, expose them as sidebar sliders so viewers can stress-test assumptions without a pipeline rerun. Reference implementation: `dashboard/teams/ai_classification_intent/scoring.py`.

The pattern:

1. Keep the formula as a **pure function** of a weights dict (or params dict). No Streamlit inside it — easy to unit-test.
2. Provide a `weight_controls()` or `param_controls()` function that renders sliders in the sidebar, persists values in `st.session_state` under a team-unique key prefix, and returns a normalized params dict.
3. At the top of each page, call `weights = weight_controls()`, then apply `recompute(df, weights)` to the loaded frame before any downstream rendering.
4. Always display a caption showing the applied (normalized) values so the viewer knows what's in effect.
5. Do not rewrite the stored artifacts — this is a view-only override.

## 11. Links

- `docs/dashboard_teams.md` — human-readable overview for students.
- `docs/agent_handoff.md` — running log of the project's state.
- `README.md` — repo-wide introduction and pipeline entry points.
