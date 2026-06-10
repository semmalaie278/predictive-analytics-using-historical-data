"""
modules/preprocessor.py
-----------------------
Handles all data cleaning and preprocessing steps:
  - Load CSV into a Pandas DataFrame
  - Report null / duplicate counts
  - Fill or drop missing values
  - Remove duplicate rows
  - Parse date columns automatically
  - Convert object columns to numeric where possible
"""

import pandas as pd
import numpy as np
from io import StringIO


# ── Public API ────────────────────────────────────────────────────────────────

def load_csv(filepath: str) -> pd.DataFrame:
    """Read a CSV file and return a raw DataFrame."""
    return pd.read_csv(filepath)


def preprocess(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Clean *df* in-place and return (cleaned_df, report).

    Report keys
    -----------
    original_shape      : (rows, cols) before cleaning
    cleaned_shape       : (rows, cols) after cleaning
    null_counts_before  : {col: count} for cols that had nulls
    duplicates_removed  : number of duplicate rows dropped
    date_columns        : list of columns converted to datetime
    numeric_conversions : list of columns converted to numeric
    fill_strategy       : strategy used for numeric null-filling
    """
    report: dict = {}

    # ── 1. Record original shape ─────────────────────────────────────────────
    report["original_shape"] = list(df.shape)

    # ── 2. Null counts before cleaning ──────────────────────────────────────
    null_before = df.isnull().sum()
    report["null_counts_before"] = {
        col: int(count) for col, count in null_before.items() if count > 0
    }

    # ── 3. Remove duplicates ─────────────────────────────────────────────────
    n_before = len(df)
    df = df.drop_duplicates()
    report["duplicates_removed"] = n_before - len(df)

    # ── 4. Auto-detect and parse date columns ────────────────────────────────
    date_cols = []
    for col in df.select_dtypes(include="object").columns:
        sample = df[col].dropna().head(20)
        converted = pd.to_datetime(sample, errors="coerce", infer_datetime_format=True)
        # If >80 % of the sample parsed successfully, treat as date
        if converted.notna().mean() > 0.8:
            try:
                df[col] = pd.to_datetime(df[col], errors="coerce", infer_datetime_format=True)
                date_cols.append(col)
            except Exception:
                pass
    report["date_columns"] = date_cols

    # ── 5. Convert remaining object columns to numeric where possible ────────
    numeric_converted = []
    for col in df.select_dtypes(include="object").columns:
        converted = pd.to_numeric(df[col], errors="coerce")
        if converted.notna().mean() > 0.5:          # majority parseable
            df[col] = converted
            numeric_converted.append(col)
    report["numeric_conversions"] = numeric_converted

    # ── 6. Fill missing values ────────────────────────────────────────────────
    #   Numeric  → median (robust to outliers)
    #   Datetime → forward-fill
    #   Object   → mode (most frequent)
    report["fill_strategy"] = "median (numeric), mode (categorical), ffill (datetime)"

    for col in df.columns:
        if df[col].isnull().sum() == 0:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = df[col].ffill().bfill()
        else:
            mode_val = df[col].mode()
            df[col] = df[col].fillna(mode_val[0] if len(mode_val) else "Unknown")

    # ── 7. Final shape ───────────────────────────────────────────────────────
    report["cleaned_shape"] = list(df.shape)

    return df.reset_index(drop=True), report


def df_preview(df: pd.DataFrame, n: int = 10) -> dict:
    """Return a JSON-serialisable preview of the first *n* rows."""
    preview = df.head(n).copy()
    # Convert datetime columns to strings for JSON serialisation
    for col in preview.select_dtypes(include=["datetime64[ns]", "datetimetz"]).columns:
        preview[col] = preview[col].astype(str)
    return {
        "columns": list(preview.columns),
        "rows": preview.replace({np.nan: None}).to_dict(orient="records"),
    }
