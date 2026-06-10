"""
modules/model_selector.py
--------------------------
Heuristic logic to determine:
  - Whether a dataset should use ARIMA (time-series) or Linear Regression
  - Which column is the date/index column (if any)
  - Which column is the prediction target
  - Which columns are features
"""

import pandas as pd


def detect_model_type(df: pd.DataFrame) -> dict:
    """
    Analyse *df* and return a recommendation dict:

    Returns
    -------
    {
        "model_type"  : "timeseries" | "regression",
        "date_col"    : str | None,
        "target_col"  : str,
        "feature_cols": [str, ...],
        "reason"      : str            # human-readable explanation
    }
    """
    # ── Step 1: look for a datetime column ───────────────────────────────────
    date_col = _find_date_column(df)

    # ── Step 2: find numeric columns ─────────────────────────────────────────
    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    if not numeric_cols:
        return {
            "model_type": "regression",
            "date_col": None,
            "target_col": "",
            "feature_cols": [],
            "reason": "No numeric columns found; defaulting to regression.",
        }

    # ── Step 3: decide model type ────────────────────────────────────────────
    if date_col is not None and len(numeric_cols) >= 1:
        # Time-series: one numeric variable indexed by time
        target_col = numeric_cols[0]        # first numeric is the series
        feature_cols = []
        model_type = "timeseries"
        reason = (
            f"Detected datetime column '{date_col}'. "
            f"Will forecast '{target_col}' using ARIMA."
        )
    else:
        # Regression: last numeric column is default target, rest are features
        target_col  = numeric_cols[-1]
        feature_cols = numeric_cols[:-1]
        model_type  = "regression"
        reason = (
            f"No datetime column detected. "
            f"Will predict '{target_col}' from {feature_cols} using Linear Regression."
        )

    return {
        "model_type"  : model_type,
        "date_col"    : date_col,
        "target_col"  : target_col,
        "feature_cols": feature_cols,
        "reason"      : reason,
    }


def get_numeric_columns(df: pd.DataFrame) -> list[str]:
    """Return all numeric column names."""
    return df.select_dtypes(include="number").columns.tolist()


# ── Internal helpers ──────────────────────────────────────────────────────────

def _find_date_column(df: pd.DataFrame) -> str | None:
    """
    Return the name of the first datetime-like column, or None if absent.
    Checks actual datetime dtype first, then object columns by name heuristic.
    """
    # 1. Already parsed datetime dtype
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            return col

    # 2. Column name heuristic (date / time / year / month / period …)
    date_keywords = {"date", "time", "year", "month", "period", "day", "week", "timestamp"}
    for col in df.columns:
        if any(kw in col.lower() for kw in date_keywords):
            # Try to parse
            try:
                parsed = pd.to_datetime(df[col], errors="coerce", infer_datetime_format=True)
                if parsed.notna().mean() > 0.7:
                    df[col] = parsed          # mutate in-place (already cleaned)
                    return col
            except Exception:
                pass

    return None
