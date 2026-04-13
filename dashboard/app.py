import streamlit as st
import pandas as pd
from pathlib import Path

st.set_page_config(page_title="AI Intent Dashboard", layout="wide")

# Import shared data loader
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_loader import load_scores

st.title("BUFN403 — AI Intent & Classification Dashboard")

st.markdown("""
This dashboard visualizes AI adoption intent across major U.S. banks, built as
part of the BUFN403 Capstone project. The system classifies public disclosures
(earnings calls, 10-K filings, press releases) into intent levels — **Exploring**,
**Committing**, **Deploying**, and **Scaling** — and scores each bank on
**Maturity**, **Breadth**, and **Momentum** dimensions.

Use the sidebar to navigate between pages.
""")

scores = load_scores(__file__)

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Banks Analyzed", len(scores))
with col2:
    st.metric("Avg Composite Score", f"{scores['Composite'].mean():.1f}")
with col3:
    top = scores.loc[scores["Composite"].idxmax()]
    st.metric("Top Bank", f"{top['Bank']} ({top['Composite']:.1f})")

st.divider()
st.subheader("Quick Snapshot")
st.dataframe(
    scores[["Rank", "Ticker", "Bank", "Composite"]].head(5),
    hide_index=True,
    use_container_width=True,
)
