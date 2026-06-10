"""
modules/timeseries.py
----------------------
ARIMA time-series forecasting pipeline:
  - Sort by date column
  - Chronological train / test split
  - Fit statsmodels ARIMA(1,1,1)
  - Evaluate: MAE, MSE, RMSE (no R² for ARIMA)
  - Forecast N future periods with 95 % confidence interval
  - Plotly chart generation
"""

import warnings
import numpy as np
import pandas as pd
import json
import plotly.graph_objects as go
from plotly.utils import PlotlyJSONEncoder
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error

from config import TEST_SIZE, ARIMA_ORDER

warnings.filterwarnings("ignore")       # suppress statsmodels convergence warnings


# ── Module-level model store ──────────────────────────────────────────────────
_ts_store: dict = {}


# ── Public API ────────────────────────────────────────────────────────────────

def train_and_evaluate(
    df: pd.DataFrame,
    date_col: str,
    target_col: str,
) -> dict:
    """
    Fit ARIMA on the target series and return metrics + chart JSON.

    Returns
    -------
    {
        "metrics"         : {mae, mse, rmse},
        "actual_vs_pred"  : str (Plotly JSON),
        "model_ready"     : True,
        "series_length"   : int,
        "train_size"      : int,
        "test_size"       : int,
    }
    """
    # ── Prepare series ────────────────────────────────────────────────────────
    ts = (
        df[[date_col, target_col]]
        .dropna()
        .sort_values(date_col)
        .set_index(date_col)[target_col]
        .astype(float)
    )

    n        = len(ts)
    n_test   = max(int(n * TEST_SIZE), 2)
    n_train  = n - n_test

    train = ts.iloc[:n_train]
    test  = ts.iloc[n_train:]

    # ── Fit ARIMA ─────────────────────────────────────────────────────────────
    model_fit = ARIMA(train, order=ARIMA_ORDER).fit()

    # ── In-sample forecast over test window ───────────────────────────────────
    forecast_res = model_fit.forecast(steps=n_test)
    y_pred = forecast_res.values
    y_true = test.values

    # ── Metrics ───────────────────────────────────────────────────────────────
    mae  = float(mean_absolute_error(y_true, y_pred))
    mse  = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))

    # ── Actual vs Predicted chart ─────────────────────────────────────────────
    test_dates = test.index.astype(str).tolist()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=train.index.astype(str).tolist(), y=train.values.tolist(),
                             mode="lines", name="Train",
                             line=dict(color="#6C63FF", width=1.5)))
    fig.add_trace(go.Scatter(x=test_dates, y=y_true.tolist(),
                             mode="lines+markers", name="Actual (Test)",
                             line=dict(color="#43E97B", width=2)))
    fig.add_trace(go.Scatter(x=test_dates, y=y_pred.tolist(),
                             mode="lines+markers", name="ARIMA Predicted",
                             line=dict(color="#FF6584", width=2, dash="dot")))
    _dark_layout(fig, "ARIMA: Actual vs Predicted", date_col, target_col)
    chart_json = json.dumps(fig, cls=PlotlyJSONEncoder)

    # ── Stash for forecast ────────────────────────────────────────────────────
    _ts_store["model_fit"]  = model_fit
    _ts_store["ts"]         = ts
    _ts_store["date_col"]   = date_col
    _ts_store["target_col"] = target_col
    _ts_store["train"]      = train

    return {
        "metrics"        : {"mae": round(mae, 4), "mse": round(mse, 4), "rmse": round(rmse, 4)},
        "actual_vs_pred" : chart_json,
        "model_ready"    : True,
        "series_length"  : n,
        "train_size"     : n_train,
        "test_size"      : n_test,
    }


def forecast(n_periods: int) -> dict:
    """
    Forecast *n_periods* future time steps beyond the training data.

    Returns
    -------
    {
        "forecast_values"     : [float, ...],
        "forecast_lower"      : [float, ...],
        "forecast_upper"      : [float, ...],
        "forecast_chart"      : str (Plotly JSON),
        "csv_rows"            : [[period, value], ...]
    }
    """
    model_fit  = _ts_store["model_fit"]
    ts         = _ts_store["ts"]
    date_col   = _ts_store["date_col"]
    target_col = _ts_store["target_col"]

    # Re-fit on full series for best forecast
    full_fit = ARIMA(ts, order=ARIMA_ORDER).fit()
    fc_res   = full_fit.get_forecast(steps=n_periods)
    fc_mean  = fc_res.predicted_mean
    fc_ci    = fc_res.conf_int(alpha=0.05)

    forecast_vals  = fc_mean.values.tolist()
    forecast_lower = fc_ci.iloc[:, 0].values.tolist()
    forecast_upper = fc_ci.iloc[:, 1].values.tolist()

    # Build future period labels
    future_labels = [f"Period +{i+1}" for i in range(n_periods)]

    # ── Forecast chart ────────────────────────────────────────────────────────
    hist_dates = ts.index.astype(str).tolist()
    fig = go.Figure()

    # Historical series
    fig.add_trace(go.Scatter(
        x=hist_dates, y=ts.values.tolist(),
        mode="lines", name="Historical",
        line=dict(color="#6C63FF", width=2)
    ))

    # Confidence band (shaded)
    fig.add_trace(go.Scatter(
        x=future_labels + future_labels[::-1],
        y=forecast_upper + forecast_lower[::-1],
        fill="toself",
        fillcolor="rgba(67,233,123,0.15)",
        line=dict(color="rgba(255,255,255,0)"),
        name="95% Confidence",
        showlegend=True,
    ))

    # Forecast line
    fig.add_trace(go.Scatter(
        x=future_labels, y=forecast_vals,
        mode="lines+markers", name="Forecast",
        line=dict(color="#43E97B", width=2, dash="dot"),
        marker=dict(size=6),
    ))

    _dark_layout(fig, f"ARIMA Forecast — Next {n_periods} Periods", "Period", target_col)
    chart_json = json.dumps(fig, cls=PlotlyJSONEncoder)

    csv_rows = [["Period", target_col, "Lower 95%", "Upper 95%"]] + [
        [future_labels[i], round(forecast_vals[i], 4),
         round(forecast_lower[i], 4), round(forecast_upper[i], 4)]
        for i in range(n_periods)
    ]

    return {
        "forecast_values" : [round(v, 4) for v in forecast_vals],
        "forecast_lower"  : [round(v, 4) for v in forecast_lower],
        "forecast_upper"  : [round(v, 4) for v in forecast_upper],
        "forecast_chart"  : chart_json,
        "csv_rows"        : csv_rows,
    }


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
