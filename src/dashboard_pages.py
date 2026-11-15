"""Page renderers; data engineering and model training stay outside the UI."""
import pandas as pd
import plotly.express as px
import streamlit as st

from src.dashboard_data import filter_dates, price_column
from src.dashboard_ui import COLORS, chart, header


def overview(raw, dates):
    header("NVDA / Historical market overview", "NVIDIA Stock Predictor",
           "Explore adjusted prices, trading activity, and the history behind the next model.")
    latest = raw.iloc[-1]
    price = price_column(raw)
    delta = None if len(raw) < 2 else f"{(latest[price] / raw[price].iloc[-2] - 1):+.2%}"
    st.caption(f"LATEST AVAILABLE TRADING DATE · {latest['Date']:%d %b %Y} · FULL DATASET METRICS")
    cols = st.columns(4)
    cols[0].metric("Latest adjusted close", f"${latest[price]:,.2f}")
    cols[1].metric("Previous trading day change", delta if delta else "Unavailable")
    cols[2].metric("Historical high · adjusted close", f"${raw[price].max():,.2f}")
    cols[3].metric("Trading days", f"{len(raw):,}")
    st.caption("Historical CSV snapshot · prices in USD · no live market feed")
    selected = filter_dates(raw, *dates)
    if selected.empty:
        st.info("No trading records in the selected date range. Choose a wider range.")
        return
    st.subheader("Price history")
    st.caption(f"SELECTED PERIOD · {dates[0]:%d %b %Y} — {dates[1]:%d %b %Y} · {len(selected):,} TRADING DAYS")
    fig = px.area(selected, x="Date", y=price, color_discrete_sequence=COLORS,
                  labels={price: "Adjusted close (USD)"})
    fig.update_traces(line_width=2, fillcolor="rgba(118,185,0,.10)")
    fig.update_xaxes(rangeslider_visible=True)
    chart(fig, "overview_price", height=420)
    st.subheader("Trading volume")
    chart(px.bar(selected, x="Date", y="Volume", color_discrete_sequence=["#44761D"],
                 labels={"Volume": "Shares traded"}), "overview_volume", height=250)
    st.subheader("Recent historical data")
    st.caption("Latest 20 trading records within the selected period · newest first")
    display = selected.tail(20).sort_values("Date", ascending=False).copy()
    display["Date"] = display["Date"].dt.date
    display = display.rename(columns={price: "Adjusted Close (USD)"})
    st.dataframe(display, hide_index=True, width="stretch")


def market_analysis(features, dates):
    header("NVDA / Market research", "Market Analysis",
           "Trend, momentum, and risk from the saved feature dataset.")
    st.caption(f"LATEST PROCESSED TRADING DATE · {features['Date'].max():%d %b %Y}")
    df = filter_dates(features, *dates)
    if df.empty:
        st.info("No processed records in this date range. Indicators begin after the initial 20-day window.")
        return
    st.caption(f"SELECTED PERIOD · {len(df):,} TRADING DAYS · saved indicators retain their full historical lookback")
    st.subheader("Price & moving averages")
    chart(px.line(df, x="Date", y=["Close", "MA_5", "MA_20"],
                  color_discrete_sequence=COLORS,
                  labels={"value": "Adjusted price (USD)", "variable": "Series"}), "moving_averages")
    left, right = st.columns(2)
    with left:
        st.subheader("Daily returns")
        chart(px.line(df, x="Date", y="Daily_Return", color_discrete_sequence=COLORS,
                      labels={"Daily_Return": "Daily return"}), "daily_returns", percent=True)
    with right:
        st.subheader("10-day rolling volatility")
        chart(px.line(df, x="Date", y="Volatility_10", color_discrete_sequence=[COLORS[1]],
                      labels={"Volatility_10": "Return standard deviation"}), "volatility", percent=True)
    st.subheader("Daily returns distribution")
    distribution = px.histogram(df, x="Daily_Return", nbins=60,
                                color_discrete_sequence=COLORS,
                                labels={"Daily_Return": "Daily return"})
    distribution.update_xaxes(tickformat=".1%")
    distribution.update_layout(yaxis_title="Trading days", hovermode="closest")
    chart(distribution, "return_distribution", height=300)
    st.subheader("Summary statistics")
    summary = df[["Close", "MA_5", "MA_20", "Daily_Return", "Volatility_10"]].copy()
    summary[["Daily_Return", "Volatility_10"]] *= 100
    summary.columns = ["Adjusted Close (USD)", "MA5 (USD)", "MA20 (USD)",
                       "Daily Return (%)", "10-day Volatility (%)"]
    st.dataframe(summary.describe().round(3), width="stretch")
    st.caption("Volatility is the sample standard deviation of 10 daily returns; it is not annualized.")
