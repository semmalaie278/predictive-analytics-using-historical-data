"""
modules/visualizer.py
----------------------
Generates the historical data overview charts used in the EDA step:
  - Historical data scatter / line chart (numeric columns)
  - Pairwise feature scatter matrix
All charts are returned as Plotly JSON strings for the frontend.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.utils import PlotlyJSONEncoder
import json


PALETTE = [
    "#6C63FF", "#FF6584", "#43E97B", "#F5AF19",
    "#00C9FF", "#FC466B", "#38F9D7", "#FAD961",
]


def historical_overview(df: pd.DataFrame, date_col: str | None = None) -> str:
    """
    Return a Plotly JSON string for a multi-series line/scatter chart of all
    numeric columns (up to 6) in *df*.  If *date_col* is provided it is used
    as the x-axis; otherwise the row index is used.
    """
    numeric_cols = df.select_dtypes(include="number").columns.tolist()[:6]
    if not numeric_cols:
        return "{}"

    x_vals = (
        df[date_col].astype(str).tolist()
        if date_col and date_col in df.columns
        else list(range(len(df)))
    )

    fig = go.Figure()
    for i, col in enumerate(numeric_cols):
        fig.add_trace(go.Scatter(
            x=x_vals,
            y=df[col].tolist(),
            mode="lines",
            name=col,
            line=dict(color=PALETTE[i % len(PALETTE)], width=1.8),
        ))

    _dark_layout(fig, "Historical Data Overview",
                 xaxis_title=date_col or "Index",
                 yaxis_title="Value")
    return json.dumps(fig, cls=PlotlyJSONEncoder)


def feature_scatter_matrix(df: pd.DataFrame, target_col: str) -> str:
    """
    Return a Plotly splom (scatter plot matrix) JSON for numeric features
    coloured by target_col quantile.  Limited to 5 columns for readability.
    """
    numeric_cols = df.select_dtypes(include="number").columns.tolist()[:5]
    if len(numeric_cols) < 2:
        return "{}"

    dims = [
        dict(label=col, values=df[col].tolist())
        for col in numeric_cols
    ]

    # Colour by target_col (if it's among numeric cols)
    color_vals = (
        df[target_col].tolist()
        if target_col in df.columns
        else list(range(len(df)))
    )

    fig = go.Figure(go.Splom(
        dimensions=dims,
        marker=dict(
            color=color_vals,
            colorscale="Viridis",
            showscale=False,
            size=3,
            opacity=0.6,
        ),
        diagonal_visible=False,
        showupperhalf=False,
    ))
    _dark_layout(fig, "Feature Scatter Matrix")
    return json.dumps(fig, cls=PlotlyJSONEncoder)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _dark_layout(fig: go.Figure, title: str = "",
                  xaxis_title: str = "", yaxis_title: str = "") -> None:
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color="#e2e8f0")),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#cbd5e1", family="Inter, sans-serif"),
        xaxis=dict(title=xaxis_title, gridcolor="rgba(255,255,255,0.07)"),
        yaxis=dict(title=yaxis_title, gridcolor="rgba(255,255,255,0.07)"),
        margin=dict(l=40, r=20, t=50, b=40),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
