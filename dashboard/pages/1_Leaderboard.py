import streamlit as st
import pandas as pd
from pathlib import Path

st.set_page_config(page_title="Leaderboard", layout="wide")

# Import shared data loader
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_loader import load_scores, load_app_categories

st.title("Leaderboard")

scores = load_scores(__file__)
app_cats = load_app_categories(__file__)

if scores.empty:
    st.warning(
        "No leaderboard data is available yet. Run the classification and scoring pipeline to generate dashboard artifacts."
    )
    st.stop()

# Filters
col_filter1, col_filter2, col_filter3 = st.columns(3)

with col_filter1:
    sort_col = st.selectbox("Sort by", ["Composite", "Maturity", "Breadth", "Momentum"], index=0)

with col_filter2:
    min_score = st.slider("Minimum Composite Score", 0.0, 100.0, 0.0, step=1.0)

with col_filter3:
    all_categories = sorted(app_cats["Category"].unique()) if not app_cats.empty else []
    selected_cats = st.multiselect("Filter by Application Category", all_categories, default=[])

# Apply filters
filtered = scores[scores["Composite"] >= min_score].copy()

if selected_cats:
    tickers_with_cats = app_cats[
        app_cats["Category"].isin(selected_cats) & (app_cats["Mention_Count"] > 0)
    ]["Ticker"].unique()
    filtered = filtered[filtered["Ticker"].isin(tickers_with_cats)]

filtered = filtered.sort_values(sort_col, ascending=False).reset_index(drop=True)
filtered["Rank"] = filtered.index + 1

score_cols = ["Maturity", "Breadth", "Momentum", "Composite"]
styled = filtered.style.background_gradient(
    subset=score_cols, cmap="YlGnBu"
).format({c: "{:.1f}" for c in score_cols})

st.dataframe(styled, hide_index=True, use_container_width=True, height=450)

st.caption(f"Showing {len(filtered)} of {len(scores)} banks")
