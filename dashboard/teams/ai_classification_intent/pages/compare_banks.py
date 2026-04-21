import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from dashboard.teams.ai_classification_intent.data_loader import load_scores, load_quarterly, load_app_categories
from dashboard.teams.ai_classification_intent.scoring import weight_controls, recompute_scores

st.title("Compare Banks")

scores = load_scores()
weights = weight_controls()
if not scores.empty:
    scores = recompute_scores(scores, weights)
quarterly = load_quarterly()
app_cats = load_app_categories()

if scores.empty:
    st.warning(
        "No comparison data is available yet. Run the classification and scoring pipeline to populate this page."
    )
    st.stop()

bank_map = {row["Ticker"]: row["Bank"] for _, row in scores.iterrows()}
selected = st.multiselect(
    "Select 2-4 banks to compare",
    options=list(bank_map.keys()),
    default=list(bank_map.keys())[:3],
    format_func=lambda t: f"{bank_map[t]} ({t})",
    max_selections=4
)

if len(selected) < 2:
    st.warning("Please select at least 2 banks to compare.")
    st.stop()

colors = px.colors.qualitative.Set1

# Overlaid Radar Charts
st.subheader("Overlaid Radar Charts")
radar_col, bar_col = st.columns(2)

with radar_col:
    fig = go.Figure()
    for i, ticker in enumerate(selected):
        bank_apps = app_cats[app_cats["Ticker"] == ticker]
        if bank_apps.empty:
            continue
        cats = bank_apps["Category"].tolist()
        vals = bank_apps["Mention_Count"].tolist()
        fig.add_trace(go.Scatterpolar(
            r=vals + [vals[0]],
            theta=cats + [cats[0]],
            fill="toself",
            name=ticker,
            line_color=colors[i % len(colors)],
            opacity=0.6
        ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True)),
        margin=dict(t=30, b=30, l=40, r=40),
        height=420,
    )
    st.plotly_chart(fig, use_container_width=True)

# Side-by-Side Intent Bars
with bar_col:
    st.subheader("Intent Level Distribution")
    latest_rows = []
    for ticker in selected:
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
                     barmode="group", color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_layout(margin=dict(t=30, b=30), height=420)
        st.plotly_chart(fig, use_container_width=True)

# Metrics Comparison Table
st.subheader("Metrics Comparison")
compare_df = scores[scores["Ticker"].isin(selected)][
    ["Ticker", "Bank", "Maturity", "Breadth", "Momentum", "Composite"]
].copy()

metric_cols = ["Maturity", "Breadth", "Momentum", "Composite"]


def highlight_leader(s):
    is_max = s == s.max()
    return ["font-weight: bold; background-color: #d4edda" if v else "" for v in is_max]


styled = compare_df.style.apply(highlight_leader, subset=metric_cols)
styled = styled.format({c: "{:.1f}" for c in metric_cols})
st.dataframe(styled, hide_index=True, use_container_width=True)
