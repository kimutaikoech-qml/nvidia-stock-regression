"""NVIDIA analytics dashboard. Run with: streamlit run app.py."""
import streamlit as st

from src.dashboard_data import (DataError, FEATURES_PATH, RAW_PATH,
                                check_feature_prices, load_dataset)
from src.dashboard_pages import market_analysis, overview
from src.model_pages import insights, predictions
from src.dashboard_ui import apply_style, sidebar_logo

PAGES = ["Overview", "Market Analysis", "ML Predictions", "Model Insights"]


def main():
    st.set_page_config(page_title="NVIDIA Stock Predictor", page_icon="📈", layout="wide")
    apply_style()
    with st.sidebar:
        sidebar_logo()
        st.caption("NVDA · STOCK PREDICTOR")
        st.divider()
        page = st.radio("Workspace", PAGES, label_visibility="collapsed")
        st.divider()
        st.caption("HISTORICAL ANALYTICS")

    # Model pages validate their own artifacts independently of the raw CSV.
    if page == "ML Predictions":
        predictions()
        return
    if page == "Model Insights":
        insights()
        return

    try:
        raw = load_dataset(RAW_PATH)
    except DataError as exc:
        st.error(str(exc))
        st.info("Download the historical CSV with: venv/bin/python src/data.py")
        return
    with st.sidebar:
        st.caption(f"Latest trading date: {raw['Date'].max():%d %b %Y}")
        chosen = st.date_input("Historical date range",
                               value=(raw['Date'].min().date(), raw['Date'].max().date()),
                               min_value=raw['Date'].min().date(), max_value=raw['Date'].max().date(),
                               key="historical_dates")
        st.caption("Adjusted prices · USD\n\nHistorical snapshot, not live quotes.")
        st.caption("Source: local Yahoo Finance CSV. Close is adjusted by the existing auto_adjust=True downloader; explicit Adj Close takes priority when present.")
    if len(chosen) != 2:
        st.info("Select both a start date and an end date to view historical analytics.")
        return
    if chosen[0] > chosen[1]:
        st.warning("The start date must be on or before the end date.")
        return
    if page == "Overview":
        overview(raw, chosen)
    else:
        try:
            features = load_dataset(FEATURES_PATH, processed=True)
            check_feature_prices(raw, features)
        except DataError as exc:
            st.error(str(exc))
            st.info("Check the processed CSV. To rebuild it when needed: venv/bin/python src/features.py")
            return
        if features['Date'].max() != raw['Date'].max():
            st.warning("Raw and processed datasets have different latest trading dates. Analysis uses the saved processed snapshot.")
        market_analysis(features, chosen)


if __name__ == "__main__":
    main()
