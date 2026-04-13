import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

st.set_page_config(page_title="Market Overview", layout="wide")

# Import shared data loader
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_loader import load_scores, load_quarterly, load_app_categories

st.title("Market Overview")

scores = load_scores(__file__)
quarterly = load_quarterly(__file__)
app_cats = load_app_categories(__file__)

# Row 1: Heatmap + Industry Intent Stack
col1, col2 = st.columns(2)

with col1:
    st.subheader("Application Category Heatmap")
    pivot = app_cats.pivot_table(index="Ticker", columns="Category",
                                 values="Mention_Count", aggfunc="sum").fillna(0)
    fig = px.imshow(
        pivot,
        labels=dict(x="Category", y="Bank", color="Mentions"),
        color_continuous_scale="YlGnBu",
        aspect="auto"
    )
    fig.update_layout(margin=dict(t=30, b=30, l=30, r=30), height=400)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Industry Intent Distribution")
    # Use all tickers from the quarterly data (not hardcoded MOCK_BANKS)
    all_tickers = quarterly["Ticker"].unique()
    latest_rows = []
    for ticker in all_tickers:
        bank_q = quarterly[quarterly["Ticker"] == ticker].sort_values(["Year", "Quarter"])
        if bank_q.empty:
            continue
        latest = bank_q.iloc[-1]
        for level in ["Exploring", "Committing", "Deploying", "Scaling"]:
            latest_rows.append(dict(
                Ticker=ticker,
                Intent=level,
                Percentage=latest[f"{level}_Pct"]
            ))
    if latest_rows:
        intent_df = pd.DataFrame(latest_rows)
        fig = px.bar(intent_df, x="Ticker", y="Percentage", color="Intent",
                     barmode="stack", color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_layout(margin=dict(t=30, b=30), height=400)
        st.plotly_chart(fig, use_container_width=True)

# Row 2: Maturity Quadrant + Top Movers
col3, col4 = st.columns(2)

with col3:
    st.subheader("Maturity Quadrant")
    fig = px.scatter(
        scores, x="Composite", y="Maturity",
        size="Breadth", text="Ticker",
        color="Ticker",
        color_discrete_sequence=px.colors.qualitative.Set1,
        size_max=45
    )
    fig.update_traces(textposition="top center")
    fig.update_layout(
        margin=dict(t=30, b=30), height=400,
        showlegend=False,
        xaxis_title="Composite Score",
        yaxis_title="Maturity Score"
    )
    # Add quadrant lines at median
    med_x = scores["Composite"].median()
    med_y = scores["Maturity"].median()
    fig.add_hline(y=med_y, line_dash="dash", line_color="gray", opacity=0.5)
    fig.add_vline(x=med_x, line_dash="dash", line_color="gray", opacity=0.5)
    st.plotly_chart(fig, use_container_width=True)

with col4:
    st.subheader("Top Movers (Momentum)")
    movers = scores.sort_values("Momentum", ascending=True).copy()
    colors_list = ["#e74c3c" if v < 0 else "#27ae60" for v in movers["Momentum"]]
    fig = go.Figure(go.Bar(
        x=movers["Momentum"],
        y=movers["Ticker"],
        orientation="h",
        marker_color=colors_list,
        text=movers["Momentum"].apply(lambda v: f"{v:+.1f}"),
        textposition="outside"
    ))
    fig.update_layout(
        margin=dict(t=30, b=30, l=60, r=60),
        height=400,
        xaxis_title="Momentum Score",
        yaxis=dict(autorange="reversed")
    )
    st.plotly_chart(fig, use_container_width=True)
