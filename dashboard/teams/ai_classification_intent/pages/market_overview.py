import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from dashboard.teams.ai_classification_intent.data_loader import load_scores, load_quarterly, load_app_categories
from dashboard.teams.ai_classification_intent.scoring import weight_controls, recompute_scores

st.title("Market Overview")

scores = load_scores()
weights = weight_controls()
if not scores.empty:
    scores = recompute_scores(scores, weights)
quarterly = load_quarterly()
app_cats = load_app_categories()

if scores.empty:
    st.warning(
        "No market overview data is available yet. Run the classification and scoring pipeline to populate this page."
    )
    st.stop()

# Row 1: Heatmap + Industry Intent Stack
col1, col2 = st.columns(2)

with col1:
    st.subheader("Application Category Heatmap")
    if app_cats.empty:
        st.info("No application-category data is available.")
    else:
        import numpy as np
        pivot = app_cats.pivot_table(index="Ticker", columns="Category",
                                     values="Mention_Count", aggfunc="sum").fillna(0)
        # Sort banks by total mentions so high-activity banks cluster together
        pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=False).index]
        # Log-scale the values so outliers don't wash out the rest
        pivot_log = np.log1p(pivot)
        fig = px.imshow(
            pivot_log,
            labels=dict(x="Category", y="Bank", color="Log Mentions"),
            color_continuous_scale="Blues",
            aspect="auto"
        )
        # Show actual counts on hover, not log values
        fig.update_traces(
            customdata=pivot.values,
            hovertemplate="Bank: %{y}<br>Category: %{x}<br>Mentions: %{customdata:.0f}<extra></extra>"
        )
        fig.update_layout(
            margin=dict(t=30, b=30, l=30, r=30),
            height=max(400, len(pivot) * 14),
            coloraxis_colorbar_title="Log(1+n)",
        )
        st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Industry Intent Distribution")
    # Use all tickers from the quarterly data (not hardcoded MOCK_BANKS)
    latest_rows = []
    for ticker in quarterly["Ticker"].unique():
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
    else:
        st.info("No quarterly intent distribution data is available.")

# Row 2: Maturity Quadrant + Top Movers
col3, col4 = st.columns(2)

with col3:
    st.subheader("Maturity Quadrant")
    fig = px.scatter(
        scores, x="Composite", y="Maturity",
        text="Ticker",
        color="Momentum",
        color_continuous_scale="RdYlGn",
        color_continuous_midpoint=0,
    )
    fig.update_traces(
        marker=dict(size=12, line=dict(width=1, color="white")),
        textposition="top center",
        textfont=dict(size=9),
    )
    # Add quadrant lines at median
    med_x = scores["Composite"].median()
    med_y = scores["Maturity"].median()
    fig.add_hline(y=med_y, line_dash="dash", line_color="gray", opacity=0.4)
    fig.add_vline(x=med_x, line_dash="dash", line_color="gray", opacity=0.4)
    # Label the four quadrants
    x_min, x_max = scores["Composite"].min(), scores["Composite"].max()
    y_min, y_max = scores["Maturity"].min(), scores["Maturity"].max()
    pad_x = (x_max - x_min) * 0.02
    pad_y = (y_max - y_min) * 0.02
    quadrant_labels = [
        (x_max - pad_x, y_max - pad_y, "Leaders", "right", "bottom"),
        (x_min + pad_x, y_max - pad_y, "Mature Niche", "left", "bottom"),
        (x_max - pad_x, y_min + pad_y, "Broad Early", "right", "top"),
        (x_min + pad_x, y_min + pad_y, "Lagging", "left", "top"),
    ]
    for qx, qy, label, xa, ya in quadrant_labels:
        fig.add_annotation(
            x=qx, y=qy, text=f"<b>{label}</b>",
            showarrow=False, font=dict(size=11, color="rgba(150,150,150,0.7)"),
            xanchor=xa, yanchor=ya,
        )
    fig.update_layout(
        margin=dict(t=30, b=30), height=400,
        xaxis_title="Composite Score",
        yaxis_title="Maturity Score",
        coloraxis_colorbar_title="Momentum",
    )
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
