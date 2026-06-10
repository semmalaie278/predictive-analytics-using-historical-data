"""
modules/regression.py
----------------------
Linear Regression model pipeline:
  - Train / test split
  - Scikit-learn LinearRegression training
  - Evaluation: MAE, MSE, RMSE, R²
  - Future prediction by extrapolating feature trends
  - Plotly chart generation (actual vs predicted, forecast)
"""

import numpy as np
import pandas as pd
import json
import plotly.graph_objects as go
from plotly.utils import PlotlyJSONEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from config import TEST_SIZE, RANDOM_STATE


# ── Public API ────────────────────────────────────────────────────────────────

def train_and_evaluate(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
) -> dict:
    """
    Train a LinearRegression model and return metrics + chart data.

    Returns
    -------
    {
        "metrics"          : {mae, mse, rmse, r2},
        "actual_vs_pred"   : str  (Plotly JSON),
        "feature_importance": [{feature, coefficient}, ...],
        "model_ready"      : True
    }
    """
    if not feature_cols:
        # Fallback: use row index as a single feature
        df = df.copy()
        df["_index"] = np.arange(len(df))
        feature_cols = ["_index"]

    X = df[feature_cols].select_dtypes(include="number").values
    y = df[target_col].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, shuffle=False
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # ── Metrics ──────────────────────────────────────────────────────────────
    mae  = float(mean_absolute_error(y_test, y_pred))
    mse  = float(mean_squared_error(y_test, y_pred))
    rmse = float(np.sqrt(mse))
    r2   = float(r2_score(y_test, y_pred))

    # ── Actual vs Predicted chart ────────────────────────────────────────────
    indices = np.arange(len(y_test))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=indices.tolist(), y=y_test.tolist(),
                             mode="lines+markers", name="Actual",
                             line=dict(color="#6C63FF", width=2)))
    fig.add_trace(go.Scatter(x=indices.tolist(), y=y_pred.tolist(),
                             mode="lines+markers", name="Predicted",
                             line=dict(color="#FF6584", width=2, dash="dot")))
    _dark_layout(fig, "Actual vs Predicted", "Test Sample Index", target_col)
    chart_json = json.dumps(fig, cls=PlotlyJSONEncoder)

    # ── Feature importance (coefficients) ────────────────────────────────────
    importance = [
        {"feature": col, "coefficient": round(float(coef), 6)}
        for col, coef in zip(feature_cols, model.coef_)
    ]

    # Stash model + metadata for later forecast calls
    _model_store["model"]        = model
    _model_store["feature_cols"] = feature_cols
    _model_store["target_col"]   = target_col
    _model_store["X"]            = X
    _model_store["y"]            = y

    return {
        "metrics": {"mae": round(mae, 4), "mse": round(mse, 4),
                    "rmse": round(rmse, 4), "r2": round(r2, 4)},
        "actual_vs_pred": chart_json,
        "feature_importance": importance,
        "model_ready": True,
    }


def forecast(n_periods: int) -> dict:
    """
    Predict *n_periods* future values by linearly extrapolating feature trends.

    Returns
    -------
    {
        "forecast_values": [float, ...],
        "forecast_chart" : str (Plotly JSON),
        "csv_rows"       : [[period, value], ...]
    }
    """
    model        = _model_store["model"]
    X            = _model_store["X"]
    y            = _model_store["y"]
    feature_cols = _model_store["feature_cols"]

    # Extrapolate each feature column linearly
    last_X = X[-1]
    step   = X[-1] - X[-2] if len(X) > 1 else np.ones(X.shape[1])
    future_X = np.array([last_X + step * (i + 1) for i in range(n_periods)])
    future_y = model.predict(future_X).tolist()

    # Historical indices + future indices
    hist_idx   = list(range(len(y)))
    future_idx = list(range(len(y), len(y) + n_periods))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist_idx, y=y.tolist(),
                             mode="lines", name="Historical",
                             line=dict(color="#6C63FF", width=2)))
    fig.add_trace(go.Scatter(x=future_idx, y=future_y,
                             mode="lines+markers", name="Forecast",
                             line=dict(color="#43E97B", width=2, dash="dot")))
    _dark_layout(fig, f"Forecast — Next {n_periods} Periods", "Index", _model_store["target_col"])
    chart_json = json.dumps(fig, cls=PlotlyJSONEncoder)

    csv_rows = [["Period", _model_store["target_col"]]] + \
               [[i + 1, round(v, 4)] for i, v in enumerate(future_y)]

    return {
        "forecast_values": [round(v, 4) for v in future_y],
        "forecast_chart" : chart_json,
        "csv_rows"       : csv_rows,
    }


# ── Module-level model store (per-process session storage) ────────────────────
_model_store: dict = {}


# ── Internal helpers ──────────────────────────────────────────────────────────

def _dark_layout(fig: go.Figure, title: str, x: str, y: str) -> None:
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color="#e2e8f0")),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#cbd5e1", family="Inter, sans-serif"),
        xaxis=dict(title=x, gridcolor="rgba(255,255,255,0.07)"),
        yaxis=dict(title=y, gridcolor="rgba(255,255,255,0.07)"),
        margin=dict(l=40, r=20, t=50, b=40),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
