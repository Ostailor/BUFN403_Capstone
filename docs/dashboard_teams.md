# Dashboard Teams

This page is for a student joining the class mid-semester. If you are a
coding agent, read `AGENTS.md` at the repo root instead — it is more
precise.

## What you are working on

BUFN403 Capstone studies how 50 U.S. banks are adopting AI, crypto,
and private credit. The class is split into six teams arranged as a
3 x 2 matrix: three sections (AI, Crypto, Private Credit), each with
two sub-teams (Classification & Intent, Risk Analysis). Every team
contributes a slice of the same shared Streamlit dashboard.

You will not build a separate app. You will ship pages into the
existing one.

## The six teams

| Section         | Classification & Intent                  | Risk Analysis            |
|-----------------|------------------------------------------|--------------------------|
| AI              | `ai_classification_intent`               | `ai_risk`                |
| Crypto          | `crypto_classification_intent`           | `crypto_risk`            |
| Private Credit  | `private_credit_classification_intent`   | `private_credit_risk`    |

The value in each cell is the team slug. You will see it in folder
names, manifest files, and the scaffolder CLI.

## Where your team's files live

Every team has the same layout:

```
dashboard/teams/<your_slug>/
  manifest.toml        <- what the dashboard knows about you
  data_loader.py       <- functions that read your data
  pages/               <- one Streamlit page per file
```

Your raw data lives outside the dashboard, in `artifacts/<your_slug>/`.
The dashboard reads from there. This keeps the Python small and the
data portable.

## The minimum path to a working team page

1. Find your team folder. It already exists — do not create a new one.
2. Drop your data into `artifacts/<your_slug>/`. A single CSV or
   Parquet is fine to start.
3. Open `dashboard/teams/<your_slug>/manifest.toml` and set
   `artifacts_dir = "artifacts/<your_slug>"` under `[data]`.
4. Run the scaffolder to add a page:

   ```
   python scripts/new_team_page.py \
     --team <your_slug> \
     --page-slug overview \
     --title "Overview"
   ```

5. Open `dashboard/teams/<your_slug>/data_loader.py` and write a
   small function that reads your file and returns a pandas
   DataFrame. Wrap it in `@st.cache_data`.
6. Import that function from the page you just scaffolded. Show
   something — a table, a metric, a chart.
7. Run the app:

   ```
   streamlit run dashboard/app.py
   ```

That is enough to be on the board. You can iterate from there.

## The one rule you must not break

If your data is missing (file not there, empty frame, bad parse),
the page must show a warning and stop. It must not invent numbers,
fall back to mock data, or display a partial chart built from
placeholders. The helper for this is `missing_data_warning` in
`dashboard.core.ui`. Use it.

The reason is simple: this is a research capstone. A convincing-looking
chart backed by fake data is worse than a clear "no data yet" notice.

## Good habits

- Each page answers one question. Name the page after the question.
- Title case for page titles.
- Units in every axis label.
- Use Streamlit's default colors unless a chart actually needs a
  specific palette.
- Commit often. Small commits are easier for your teammates and
  graders to follow.

## How to look at a good example

The AI Classification & Intent team (Aditya Dabeer's team) has
already built out its pages. Open:

- `dashboard/teams/ai_classification_intent/manifest.toml`
- `dashboard/teams/ai_classification_intent/data_loader.py`
- `dashboard/teams/ai_classification_intent/pages/`
- `docs/how_scoring_works.md`

Treat those as templates. When you are stuck, copy the structure,
not the content.

## Where to ask questions

- For framework questions (how the app finds your team, how pages
  are registered): read `AGENTS.md` at the repo root.
- For project history and what has already been decided: read
  `docs/agent_handoff.md`.
- For pipeline questions (how the AI classification data was built):
  read `README.md` and `docs/how_scoring_works.md`.
- For anything else: ask the team lead for your section.
