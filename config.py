"""
config.py
---------
Central configuration for the Predictive Analytics application.
"""

import os

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR  = os.path.join(BASE_DIR, "uploads")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
SAMPLE_DIR  = os.path.join(BASE_DIR, "sample_data")

# Create directories if they don't exist
for _dir in (UPLOAD_DIR, RESULTS_DIR, SAMPLE_DIR):
    os.makedirs(_dir, exist_ok=True)

# ── Upload limits ─────────────────────────────────────────────────────────────
MAX_CONTENT_LENGTH = 50 * 1024 * 1024   # 50 MB
ALLOWED_EXTENSIONS = {"csv"}

# ── Model defaults ────────────────────────────────────────────────────────────
TEST_SIZE          = 0.20               # 80/20 train-test split
ARIMA_ORDER        = (1, 1, 1)          # Default ARIMA (p,d,q)
RANDOM_STATE       = 42

# ── Flask secret ─────────────────────────────────────────────────────────────
SECRET_KEY = "predictive-analytics-secret-key-2024"
