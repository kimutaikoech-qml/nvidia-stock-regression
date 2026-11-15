"""Shared training/inference contract; order is part of the saved model API."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data/processed/nvda_features.csv"
MODEL_PATH = ROOT / "models/linear_regression.pkl"
METRICS_PATH = ROOT / "results/metrics.json"
PREDICTIONS_PATH = ROOT / "results/predictions.csv"
FEATURES = ["Close", "Volume", "Daily_Return", "MA_5", "MA_20", "Volatility_10"]
ARTIFACT_SCHEMA_VERSION = 3
EVALUATION_NOTE = (
    "The chronological test set was already inspected during numerical diagnostics. "
    "These results are a diagnostic comparison, not a fresh untouched holdout. "
    "No model parameters were optimized against this test set."
)
