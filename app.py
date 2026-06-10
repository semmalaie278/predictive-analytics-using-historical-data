"""
app.py
------
Flask application entry point for the Predictive Analytics Dashboard.

Routes
------
GET  /                     → Serve the single-page UI
POST /api/upload           → Upload & validate CSV file
POST /api/preprocess       → Clean data, return report + preview
POST /api/eda              → EDA summary stats + charts
POST /api/train            → Auto-detect model, train, evaluate
POST /api/forecast         → Predict N future periods
GET  /api/download/<fname> → Download prediction CSV
GET  /api/samples          → List bundled sample datasets
POST /api/load_sample      → Load a bundled sample dataset
"""

import os
import csv
import uuid
import json

import numpy as np
import pandas as pd
from flask import (
    Flask,
    request,
    jsonify,
    render_template,
    send_from_directory,
    abort,
)
from werkzeug.utils import secure_filename

import config
from modules import preprocessor, eda, model_selector, regression, timeseries, visualizer

# ── Flask setup ───────────────────────────────────────────────────────────────
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = config.MAX_CONTENT_LENGTH
app.config["SECRET_KEY"]         = config.SECRET_KEY

# ── In-memory session store (single-user, per-process) ────────────────────────
# Stores the current cleaned DataFrame and detected model info
_session: dict = {}


# ═════════════════════════════════════════════════════════════════════════════
# Helper utilities
# ═════════════════════════════════════════════════════════════════════════════

def _allowed_file(filename: str) -> bool:
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in config.ALLOWED_EXTENSIONS
    )


def _df_to_store(df: pd.DataFrame) -> None:
    """Pickle the DataFrame into our session dict."""
    _session["df"] = df


def _df_from_store() -> pd.DataFrame | None:
    return _session.get("df")


# ═════════════════════════════════════════════════════════════════════════════
# Routes
# ═════════════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    """Serve the main SPA."""
    return render_template("index.html")


# ── Upload ────────────────────────────────────────────────────────────────────

@app.route("/api/upload", methods=["POST"])
def upload():
    """
    Accept a CSV file upload.  Validates the file, saves it temporarily,
    and returns basic info (filename, raw shape, column names).
    """
    if "file" not in request.files:
        return jsonify({"error": "No file part in request."}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected."}), 400

    if not _allowed_file(file.filename):
        return jsonify({"error": "Only CSV files are supported."}), 400

    filename  = secure_filename(file.filename)
    save_path = os.path.join(config.UPLOAD_DIR, filename)
    file.save(save_path)

    try:
        df_raw = preprocessor.load_csv(save_path)
    except Exception as e:
        return jsonify({"error": f"Could not read CSV: {e}"}), 400

    _session["raw_path"] = save_path
    _session["filename"] = filename

    return jsonify({
        "filename"  : filename,
        "raw_shape" : list(df_raw.shape),
        "columns"   : list(df_raw.columns),
        "preview"   : preprocessor.df_preview(df_raw, n=5),
    })


# ── Sample datasets ───────────────────────────────────────────────────────────

@app.route("/api/samples", methods=["GET"])
def list_samples():
    """Return the list of bundled sample CSV files."""
    files = [f for f in os.listdir(config.SAMPLE_DIR) if f.endswith(".csv")]
    return jsonify({"samples": files})


@app.route("/api/load_sample", methods=["POST"])
def load_sample():
    """Load one of the bundled sample datasets as if the user uploaded it."""
    data = request.get_json(silent=True) or {}
    name = data.get("name", "")
    path = os.path.join(config.SAMPLE_DIR, secure_filename(name))

    if not os.path.exists(path):
        return jsonify({"error": "Sample not found."}), 404

    try:
        df_raw = preprocessor.load_csv(path)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

    _session["raw_path"] = path
    _session["filename"] = name

    return jsonify({
        "filename"  : name,
        "raw_shape" : list(df_raw.shape),
        "columns"   : list(df_raw.columns),
        "preview"   : preprocessor.df_preview(df_raw, n=5),
    })


# ── Preprocess ────────────────────────────────────────────────────────────────

@app.route("/api/preprocess", methods=["POST"])
def preprocess_data():
    """
    Clean the uploaded dataset and return:
      - Preprocessing report
      - Cleaned data preview (first 10 rows)
      - Column dtypes
    """
    raw_path = _session.get("raw_path")
    if not raw_path:
        return jsonify({"error": "No file uploaded yet."}), 400

    try:
        df_raw              = preprocessor.load_csv(raw_path)
        df_clean, report    = preprocessor.preprocess(df_raw)
        _df_to_store(df_clean)
        preview             = preprocessor.df_preview(df_clean, n=10)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify({
        "report" : report,
        "preview": preview,
    })


# ── EDA ───────────────────────────────────────────────────────────────────────

@app.route("/api/eda", methods=["POST"])
def run_eda():
    """
    Return:
      - Summary statistics
      - Correlation heatmap (Plotly JSON)
      - Up to 8 distribution histograms (Plotly JSON list)
      - Historical overview chart (Plotly JSON)
      - Model-type recommendation
    """
    df = _df_from_store()
    if df is None:
        return jsonify({"error": "Data not preprocessed yet."}), 400

    try:
        stats        = eda.summary_stats(df)
        heatmap      = eda.correlation_heatmap(df)
        distributions = eda.distribution_plots(df)
        model_info   = model_selector.detect_model_type(df)
        overview     = visualizer.historical_overview(df, model_info.get("date_col"))

        # Store model info for the train step
        _session["model_info"] = model_info
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify({
        "stats"         : stats,
        "heatmap"       : heatmap,
        "distributions" : distributions,
        "overview"      : overview,
        "model_info"    : model_info,
    })


# ── Train ─────────────────────────────────────────────────────────────────────

@app.route("/api/train", methods=["POST"])
def train():
    """
    Train the auto-selected model (regression or ARIMA).
    Accepts optional JSON body to override target_col / feature_cols.

    Returns metrics, actual-vs-predicted chart, and model metadata.
    """
    df = _df_from_store()
    if df is None:
        return jsonify({"error": "Data not preprocessed yet."}), 400

    model_info = _session.get("model_info")
    if not model_info:
        model_info = model_selector.detect_model_type(df)
        _session["model_info"] = model_info

    # Allow the user to override target/feature columns via POST body
    body = request.get_json(silent=True) or {}
    if "target_col" in body:
        model_info["target_col"] = body["target_col"]
    if "feature_cols" in body:
        model_info["feature_cols"] = body["feature_cols"]
    if "model_type" in body:
        model_info["model_type"] = body["model_type"]

    try:
        if model_info["model_type"] == "timeseries":
            result = timeseries.train_and_evaluate(
                df,
                date_col   = model_info["date_col"],
                target_col = model_info["target_col"],
            )
        else:
            result = regression.train_and_evaluate(
                df,
                feature_cols = model_info["feature_cols"],
                target_col   = model_info["target_col"],
            )
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    result["model_type"] = model_info["model_type"]
    result["target_col"] = model_info["target_col"]
    return jsonify(result)


# ── Forecast ──────────────────────────────────────────────────────────────────

@app.route("/api/forecast", methods=["POST"])
def forecast():
    """
    Predict *n_periods* future values.  Saves results as a CSV file and
    returns the chart JSON + a download token.
    """
    body      = request.get_json(silent=True) or {}
    n_periods = int(body.get("n_periods", 10))
    if n_periods < 1 or n_periods > 500:
        return jsonify({"error": "n_periods must be between 1 and 500."}), 400

    model_info = _session.get("model_info", {})
    model_type = model_info.get("model_type", "regression")

    try:
        if model_type == "timeseries":
            result = timeseries.forecast(n_periods)
        else:
            result = regression.forecast(n_periods)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    # ── Save prediction CSV ───────────────────────────────────────────────────
    token    = uuid.uuid4().hex[:12]
    csv_name = f"predictions_{token}.csv"
    csv_path = os.path.join(config.RESULTS_DIR, csv_name)

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        for row in result["csv_rows"]:
            writer.writerow(row)

    result["download_token"] = csv_name
    return jsonify(result)


# ── Download ──────────────────────────────────────────────────────────────────

@app.route("/api/download/<filename>")
def download(filename: str):
    """Serve a previously generated prediction CSV."""
    safe = secure_filename(filename)
    full = os.path.join(config.RESULTS_DIR, safe)
    if not os.path.exists(full):
        abort(404)
    return send_from_directory(config.RESULTS_DIR, safe, as_attachment=True)


# ── Numeric column list ───────────────────────────────────────────────────────

@app.route("/api/columns", methods=["GET"])
def get_columns():
    """Return numeric columns of the current dataset (for target picker UI)."""
    df = _df_from_store()
    if df is None:
        return jsonify({"columns": []})
    return jsonify({
        "columns"     : list(df.columns),
        "numeric_cols": model_selector.get_numeric_columns(df),
    })


# ═════════════════════════════════════════════════════════════════════════════
# Entry point
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("🚀  Predictive Analytics Dashboard running at http://localhost:5000")
    app.run(debug=True, port=5000)
