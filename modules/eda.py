"""
modules/eda.py
--------------
Exploratory Data Analysis utilities:
  - Summary statistics (describe + dtypes + null counts)
  - Correlation matrix data
  - Distribution histograms per numeric column (Plotly JSON)
  - Time-series line chart for datetime-indexed data
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.utils import PlotlyJSONEncoder
import json


# ── Colour palette (matches our dark UI) ─────────────────────────────────────
PALETTE = [
    "#6C63FF", "#FF6584", "#43E97B", "#F5AF19",
    "#00C9FF", "#FC466B", "#38F9D7", "#FAD961",
]


def summary_stats(df: pd.DataFrame) -> dict:
    """
    Return a JSON-serialisable summary of the DataFrame.

    Returns
    -------
    dict with keys:
        shape           : [rows, cols]
        dtypes          : {col: dtype_str}
        null_counts     : {col: count}
        numeric_summary : list of row-dicts from describe()
        categorical_summary : {col: {value: count, ...} top-5}
    """
    numeric_df  = df.select_dtypes(include="number")
    cat_df      = df.select_dtypes(include="object")

    # describe() for numerics
    if not numeric_df.empty:
        desc = numeric_df.describe().T.reset_index()
        desc.columns = ["column"] + list(desc.columns[1:])
        # Round floats for readability
        num_summary = desc.round(4).replace({np.nan: None}).to_dict(orient="records")
    else:
        num_summary = []

    # Top-5 value counts per categorical column
    cat_summary = {}
    for col in cat_df.columns:
        vc = df[col].value_counts().head(5)
        cat_summary[col] = vc.to_dict()

    return {
        "shape": list(df.shape),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "null_counts": {col: int(v) for col, v in df.isnull().sum().items() if v > 0},
        "numeric_summary": num_summary,
        "categorical_summary": cat_summary,
    }


def correlation_heatmap(df: pd.DataFrame) -> str:
    """Return a Plotly heatmap of numeric column correlations as JSON."""
    numeric_df = df.select_dtypes(include="number")
    if numeric_df.shape[1] < 2:
        return "{}"

    corr = numeric_df.corr().round(3)
    fig = go.Figure(
        data=go.Heatmap(
            z=corr.values.tolist(),
            x=list(corr.columns),
            y=list(corr.index),
            colorscale="Viridis",
            zmin=-1,
            zmax=1,
            text=corr.values.round(2).tolist(),
            texttemplate="%{text}",
            showscale=True,
            hoverongaps=False,
        )
    )
    _apply_dark_layout(fig, title="Feature Correlation Heatmap")
    return json.dumps(fig, cls=PlotlyJSONEncoder)


def distribution_plots(df: pd.DataFrame) -> list[str]:
    """
    Return a list of Plotly histogram JSON strings — one per numeric column.
    Limited to the first 8 columns to keep the UI manageable.
    """
    numeric_cols = df.select_dtypes(include="number").columns.tolist()[:8]
    charts = []
    for i, col in enumerate(numeric_cols):
        color = PALETTE[i % len(PALETTE)]
        fig = go.Figure(
            go.Histogram(
                x=df[col].dropna(),
                name=col,
                marker_color=color,
                opacity=0.85,
                nbinsx=30,
            )
        )
        _apply_dark_layout(fig, title=f"Distribution — {col}", xaxis_title=col, yaxis_title="Count")
        charts.append(json.dumps(fig, cls=PlotlyJSONEncoder))
    return charts


def timeseries_chart(df: pd.DataFrame, date_col: str, value_col: str) -> str:
    """Return a Plotly line chart JSON for a time-series column."""
    series = df[[date_col, value_col]].dropna().sort_values(date_col)
    fig = go.Figure(
        go.Scatter(
            x=series[date_col].astype(str),
            y=series[value_col],
            mode="lines+markers",
            name=value_col,
            line=dict(color="#6C63FF", width=2),
            marker=dict(size=4),
        )
    )
    _apply_dark_layout(fig, title=f"{value_col} Over Time", xaxis_title=date_col, yaxis_title=value_col)
    return json.dumps(fig, cls=PlotlyJSONEncoder)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _apply_dark_layout(fig: go.Figure, title: str = "",
                        xaxis_title: str = "", yaxis_title: str = "") -> None:
    """Apply a consistent dark-theme layout to a Plotly figure."""
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color="#e2e8f0")),
        paper_bgcolor="rgba(15,15,30,0)",
        plot_bgcolor="rgba(15,15,30,0)",
        font=dict(color="#cbd5e1", family="Inter, sans-serif"),
        xaxis=dict(
            title=xaxis_title,
            gridcolor="rgba(255,255,255,0.07)",
            zerolinecolor="rgba(255,255,255,0.1)",
        ),
        yaxis=dict(
            title=yaxis_title,
            gridcolor="rgba(255,255,255,0.07)",
            zerolinecolor="rgba(255,255,255,0.1)",
        ),
        margin=dict(l=40, r=20, t=50, b=40),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#cbd5e1")),
    )
