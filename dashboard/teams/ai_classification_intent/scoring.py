"""View-side composite score overrides for the AI Classification & Intent team.

This module is kept free of Streamlit imports at module scope so the pure
scoring helpers (``normalize_weights``, ``recompute_scores``,
``format_weights_caption``) can be unit-tested without a Streamlit runtime.
Only :func:`weight_controls` imports ``streamlit`` lazily.
"""

from __future__ import annotations

import pandas as pd

DEFAULT_WEIGHTS: dict[str, float] = {"maturity": 0.50, "breadth": 0.35, "momentum": 0.15}

_REQUIRED_INPUT_COLUMNS: tuple[str, ...] = ("Maturity", "Breadth", "Momentum")
_OUTPUT_COLUMNS: tuple[str, ...] = (
    "Ticker",
    "Bank",
    "Maturity",
    "Breadth",
    "Momentum",
    "Composite",
    "Rank",
)


def normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    """Return weights summing to 1.0.

    If the supplied mapping is missing any of the three canonical keys
    (``maturity``, ``breadth``, ``momentum``) or the total weight is zero
    (or negative), fall back to :data:`DEFAULT_WEIGHTS`.
    """

    keys = ("maturity", "breadth", "momentum")
    try:
        values = {k: float(weights.get(k, 0.0)) for k in keys}
    except (TypeError, ValueError):
        return dict(DEFAULT_WEIGHTS)

    total = sum(values.values())
    if total <= 0:
        return dict(DEFAULT_WEIGHTS)

    return {k: v / total for k, v in values.items()}


def recompute_scores(
    scores_df: pd.DataFrame, weights: dict[str, float]
) -> pd.DataFrame:
    """Return a new frame with ``Composite`` and ``Rank`` recomputed.

    Formula::

        Composite = w_m * Maturity
                  + w_b * Breadth
                  + w_mo * ((Momentum + 100) / 2)

    ...where the weights are first run through :func:`normalize_weights`.
    Ranks are 1..N by descending ``Composite``, with ties broken by ascending
    ``Ticker``. Empty input or missing required columns returns the input
    unchanged (still a :class:`~pandas.DataFrame`).
    """

    if not isinstance(scores_df, pd.DataFrame):
        raise TypeError("scores_df must be a pandas DataFrame")

    if scores_df.empty:
        return scores_df.copy()

    if any(col not in scores_df.columns for col in _REQUIRED_INPUT_COLUMNS):
        return scores_df.copy()

    normalized = normalize_weights(weights)
    w_m = normalized["maturity"]
    w_b = normalized["breadth"]
    w_mo = normalized["momentum"]

    out = scores_df.copy()
    momentum_normalized = (out["Momentum"].astype(float) + 100.0) / 2.0
    composite = (
        w_m * out["Maturity"].astype(float)
        + w_b * out["Breadth"].astype(float)
        + w_mo * momentum_normalized
    )
    out["Composite"] = composite.round(2)

    # Rank: 1..N by descending Composite, ties broken ascending by Ticker.
    if "Ticker" in out.columns:
        sort_cols = ["Composite", "Ticker"]
        ascending = [False, True]
    else:
        sort_cols = ["Composite"]
        ascending = [False]

    ordered_index = out.sort_values(sort_cols, ascending=ascending).index
    rank_map = {idx: rank for rank, idx in enumerate(ordered_index, start=1)}
    out["Rank"] = out.index.map(rank_map).astype(int)

    return out


def format_weights_caption(weights: dict[str, float]) -> str:
    """Return e.g. ``'Applied weights - Maturity 50% ... Breadth 35% ... Momentum 15%'``."""

    normalized = normalize_weights(weights)
    m_pct = round(normalized["maturity"] * 100)
    b_pct = round(normalized["breadth"] * 100)
    mo_pct = round(normalized["momentum"] * 100)
    return (
        "Applied weights \u2014 "
        f"Maturity {m_pct}% \u00b7 "
        f"Breadth {b_pct}% \u00b7 "
        f"Momentum {mo_pct}%"
    )


def _reset_weights_callback(maturity_key: str, breadth_key: str) -> None:
    """Streamlit on_click callback.

    Runs before widgets instantiate on the next rerun, so assigning
    session_state here is safe even when the sliders exist.
    """

    import streamlit as st

    st.session_state[maturity_key] = int(round(DEFAULT_WEIGHTS["maturity"] * 100))
    st.session_state[breadth_key] = int(round(DEFAULT_WEIGHTS["breadth"] * 100))


def weight_controls(key_prefix: str = "ai_ci") -> dict[str, float]:
    """Render the AI Classification composite-weight controls and return normalized weights.

    Two sliders (Maturity and Breadth) and a derived Momentum metric, so the
    three weights always sum to 100 by construction. A Reset button restores
    the canonical 50 / 35 / 15 defaults via an ``on_click`` callback (required
    so session_state writes happen before the sliders instantiate).
    """

    import streamlit as st

    maturity_key = f"{key_prefix}_weight_maturity"
    breadth_key = f"{key_prefix}_weight_breadth"

    default_maturity = int(round(DEFAULT_WEIGHTS["maturity"] * 100))
    default_breadth = int(round(DEFAULT_WEIGHTS["breadth"] * 100))

    if maturity_key not in st.session_state:
        st.session_state[maturity_key] = default_maturity
    if breadth_key not in st.session_state:
        st.session_state[breadth_key] = default_breadth

    with st.sidebar.expander("AI Classification — Composite Weights", expanded=True):
        st.caption(
            "Applies only to the AI Classification & Intent scores. "
            "Weights always sum to 100%."
        )

        st.slider(
            "Maturity (%)",
            min_value=0,
            max_value=100,
            step=1,
            key=maturity_key,
        )

        maturity_pct = int(st.session_state[maturity_key])
        breadth_max = 100 - maturity_pct

        # Clamp the stored breadth value BEFORE the breadth slider instantiates —
        # Streamlit forbids writes to session_state for a widget key after that
        # widget has been rendered in the current run.
        if int(st.session_state[breadth_key]) > breadth_max:
            st.session_state[breadth_key] = breadth_max

        st.slider(
            "Breadth (%)",
            min_value=0,
            max_value=breadth_max,
            step=1,
            key=breadth_key,
        )

        breadth_pct = int(st.session_state[breadth_key])
        momentum_pct = 100 - maturity_pct - breadth_pct
        st.metric("Momentum (%)", momentum_pct, help="Derived so the three weights sum to 100%.")

        st.button(
            "Reset to defaults",
            key=f"{key_prefix}_weight_reset",
            on_click=_reset_weights_callback,
            args=(maturity_key, breadth_key),
        )

        weights = {
            "maturity": maturity_pct / 100.0,
            "breadth": breadth_pct / 100.0,
            "momentum": momentum_pct / 100.0,
        }
        st.caption(format_weights_caption(weights))

    return weights
