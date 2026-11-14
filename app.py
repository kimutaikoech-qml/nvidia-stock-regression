
import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

st.set_page_config(
    page_title="NVIDIA Stock Predictor",
    page_icon="📈",
    layout="wide"
)

st.title("NVIDIA Stock Price Prediction")
st.caption("Linear Regression Machine Learning Project")

DATA_PATH = Path("data/raw/nvda.csv")

if not DATA_PATH.exists():
    st.error(
        "Dataset not found. Run "
        "'python src/download_data.py' first."
    )
    st.stop()

# Load the saved data
df = pd.read_csv(
    DATA_PATH,
    parse_dates=["Date"]
)

df = df.sort_values("Date")

# Dashboard metrics
latest_price = df["Close"].iloc[-1]
previous_price = df["Close"].iloc[-2]

change = (
    (latest_price - previous_price)
    / previous_price
) * 100

col1, col2, col3 = st.columns(3)

col1.metric(
    "Latest Closing Price",
    f"${latest_price:.2f}",
    f"{change:.2f}%"
)

col2.metric(
    "Highest Adjusted Close",
    f"${df['Close'].max():.2f}"
)

col3.metric(
    "Trading Days",
    f"{len(df):,}"
)

# Historical price chart
st.subheader("NVIDIA Historical Stock Price")

fig = px.line(
    df,
    x="Date",
    y="Close",
    title="NVDA Adjusted Closing Price"
)

fig.update_layout(
    xaxis_title="Date",
    yaxis_title="Price (USD)",
    hovermode="x unified"
)

st.plotly_chart(fig, use_container_width=True)

# Volume chart
st.subheader("Trading Volume")

volume_fig = px.bar(
    df,
    x="Date",
    y="Volume",
    title="NVDA Daily Trading Volume"
)

st.plotly_chart(
    volume_fig,
    use_container_width=True
)

# Dataset preview
st.subheader("Historical Dataset")

st.dataframe(
    df.tail(20),
    use_container_width=True
)

# ====================================
# STEP 3: EXPLORATORY DATA ANALYSIS
# ====================================

st.divider()
st.header("Exploratory Data Analysis")

FEATURES_PATH = Path("data/processed/nvda_features.csv")

if FEATURES_PATH.exists():

    features_df = pd.read_csv(
        FEATURES_PATH,
        parse_dates=["Date"]
    )

    # Moving averages
    st.subheader("NVIDIA Moving Averages")

    fig_ma = px.line(
        features_df,
        x="Date",
        y=["Close", "MA_5", "MA_20"],
        title="NVIDIA Price vs Moving Averages"
    )

    st.plotly_chart(
        fig_ma,
        use_container_width=True
    )

    # Daily returns
    st.subheader("Daily Stock Returns")

    fig_returns = px.line(
        features_df,
        x="Date",
        y="Daily_Return",
        title="NVIDIA Daily Returns"
    )

    st.plotly_chart(
        fig_returns,
        use_container_width=True
    )

    # Volatility
    st.subheader("10-Day Rolling Volatility")

    fig_vol = px.line(
        features_df,
        x="Date",
        y="Volatility_10",
        title="NVIDIA Historical Volatility"
    )

    st.plotly_chart(
        fig_vol,
        use_container_width=True
    )

else:
    st.warning(
        "Processed data not found. "
        "Run python src/features.py first."
    )
