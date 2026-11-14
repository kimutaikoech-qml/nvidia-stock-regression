"""Read-only, validated CSV access. Cache entries follow file changes."""
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data/raw/nvda.csv"
FEATURES_PATH = ROOT / "data/processed/nvda_features.csv"
INDICATORS = ["MA_5", "MA_20", "Daily_Return", "Volatility_10"]


class DataError(ValueError):
    """A dataset is unavailable or cannot safely be displayed."""


@st.cache_data(show_spinner=False)
def _read_csv(path, modified_ns, size, processed):
    # mtime and size participate in the cache key so refreshed CSVs reload.
    try:
        df = pd.read_csv(path)
        price = "Adj Close" if "Adj Close" in df else "Close"
        required = ["Date", price, "Open", "High", "Low", "Close", "Volume"]
        if processed:
            required += INDICATORS
        missing = set(required) - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns: {', '.join(sorted(missing))}")
        if df.empty:
            raise ValueError("No trading records found.")
        df["Date"] = pd.to_datetime(df["Date"], errors="raise")
        if df["Date"].isna().any() or df["Date"].duplicated().any():
            raise ValueError("Trading dates must be present and unique.")
        numeric = list(dict.fromkeys(c for c in required if c != "Date"))
        for col in numeric:
            df[col] = pd.to_numeric(df[col], errors="raise")
        if not np.isfinite(df[numeric].to_numpy(dtype=float)).all():
            raise ValueError("Numeric data contains missing or infinite values.")
        if (df[[price, "Open", "High", "Low", "Close"]] <= 0).any().any():
            raise ValueError("Prices must be positive.")
        if (df["Volume"] < 0).any():
            raise ValueError("Trading volume cannot be negative.")
        if processed and (df["Volatility_10"] < 0).any():
            raise ValueError("Volatility cannot be negative.")
        return df.sort_values("Date").reset_index(drop=True)
    except (OSError, ValueError, TypeError) as exc:
        raise DataError(f"{Path(path).name}: {exc}") from exc


def load_dataset(path, processed=False):
    path = Path(path)
    try:
        info = path.stat()
    except OSError as exc:
        raise DataError(f"Cannot access {path.name}. Check that the CSV exists and is readable.") from exc
    return _read_csv(str(path), info.st_mtime_ns, info.st_size, processed)


def price_column(df):
    # src/data.py downloads with auto_adjust=True; its Close is adjusted.
    # Prefer an explicit Adj Close if the source format later includes it.
    return "Adj Close" if "Adj Close" in df else "Close"


def filter_dates(df, start, end):
    return df.loc[df["Date"].dt.date.between(start, end)].copy()


def check_feature_prices(raw, features):
    """Prevent showing saved indicators against a different price series."""
    overlap = features.merge(raw[["Date", price_column(raw)]].rename(
        columns={price_column(raw): "Raw_Adjusted"}), on="Date", how="inner")
    if overlap.empty or not np.allclose(overlap["Close"], overlap["Raw_Adjusted"], rtol=1e-6):
        raise DataError("Processed closing prices do not match the adjusted raw prices. "
                        "Review the CSV sources before using Market Analysis.")
