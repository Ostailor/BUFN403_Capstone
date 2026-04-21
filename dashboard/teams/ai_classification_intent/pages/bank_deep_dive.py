import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from dashboard.teams.ai_classification_intent.data_loader import load_scores, load_quarterly, load_app_categories, load_classifications
from dashboard.teams.ai_classification_intent.scoring import weight_controls, recompute_scores

st.title("Bank Deep Dive")

scores = load_scores()
weights = weight_controls()
if not scores.empty:
    scores = recompute_scores(scores, weights)
quarterly = load_quarterly()
app_cats = load_app_categories()
classifications = load_classifications()

if scores.empty:
    st.warning(
        "No bank score data is available yet. Run the classification and scoring pipeline to populate this page."
    )
    st.stop()

bank_options = {row["Ticker"]: row["Bank"] for _, row in scores.iterrows()}
selected_ticker = st.selectbox(
    "Select a Bank",
    options=list(bank_options.keys()),
    format_func=lambda t: f"{bank_options[t]} ({t})"
)

bank_name = bank_options[selected_ticker]
st.subheader(bank_name)

# Show bank's composite scores
bank_scores = scores[scores["Ticker"] == selected_ticker].iloc[0]
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Rank", f"#{int(bank_scores['Rank'])}")
m2.metric("Composite", f"{bank_scores['Composite']:.1f}")
m3.metric("Maturity", f"{bank_scores['Maturity']:.1f}")
m4.metric("Breadth", f"{bank_scores['Breadth']:.1f}")
m5.metric("Momentum", f"{bank_scores['Momentum']:+.1f}")

st.divider()

# Row 1: Intent Distribution + Application Radar
chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    st.subheader("Intent Distribution")
    bank_q = quarterly[quarterly["Ticker"] == selected_ticker].copy()
    if not bank_q.empty:
        latest = bank_q.sort_values(["Year", "Quarter"]).iloc[-1]
        intent_data = pd.DataFrame({
            "Intent Level": ["Exploring", "Committing", "Deploying", "Scaling"],
            "Percentage": [latest["Exploring_Pct"], latest["Committing_Pct"],
                           latest["Deploying_Pct"], latest["Scaling_Pct"]]
        })
        fig = px.pie(intent_data, names="Intent Level", values="Percentage",
                     hole=0.45, color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_layout(margin=dict(t=20, b=20, l=20, r=20), height=350)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No quarterly data available.")

with chart_col2:
    st.subheader("Application Radar")
    bank_apps = app_cats[app_cats["Ticker"] == selected_ticker]
    if not bank_apps.empty:
        import math
        all_categories = ["GenAI / LLMs", "Predictive ML", "NLP / Text",
                          "Computer Vision", "RPA / Automation", "Fraud / Risk Models"]
        # Ensure all 6 categories are present, fill missing with 0
        cat_data = {row["Category"]: row["Mention_Count"] for _, row in bank_apps.iterrows()}
        cats = all_categories
        counts = [cat_data.get(c, 0) for c in cats]
        # Square root scale so small values remain visible alongside large ones
        vals = [round(math.sqrt(c), 2) for c in counts]
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=vals + [vals[0]],
            theta=cats + [cats[0]],
            fill="toself",
            name=selected_ticker,
            line_color="steelblue",
            text=[f"{c} mentions" for c in counts] + [f"{counts[0]} mentions"],
            hovertemplate="%{theta}<br>%{text}<extra></extra>",
        ))
        fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=False),
                angularaxis=dict(tickfont=dict(size=11)),
            ),
            margin=dict(t=40, b=40, l=80, r=80),
            height=400,
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No application category data available.")

# Row 2: Quarterly Timeline
st.subheader("Quarterly Timeline")
if not bank_q.empty:
    bank_q = bank_q.sort_values(["Year", "Quarter"]).copy()
    bank_q["Period"] = bank_q["Year"].astype(str) + " Q" + bank_q["Quarter"].astype(str)
    fig = px.line(bank_q, x="Period", y="Maturity_Score",
                  markers=True, labels={"Maturity_Score": "Maturity Score"})
    fig.update_layout(margin=dict(t=20, b=20), height=300)
    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("No quarterly data available.")

# Row 3: Evidence Table
st.subheader("Evidence Table")
bank_class = classifications[classifications["ticker"] == selected_ticker]
if not bank_class.empty:
    display_cols = ["chunk_id", "source_type", "period_year", "period_quarter",
                    "intent_label", "confidence", "evidence_snippet"]
    available_cols = [c for c in display_cols if c in bank_class.columns]
    st.dataframe(
        bank_class[available_cols].head(10),
        hide_index=True,
        use_container_width=True
    )
else:
    st.info("No classified chunks found for this bank.")
