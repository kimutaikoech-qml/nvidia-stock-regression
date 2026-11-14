
import pandas as pd
from pathlib import Path

DATA_PATH = Path("data/raw/nvda.csv")
OUTPUT_PATH = Path("data/processed/nvda_features.csv")


def create_features():
    # Load historical NVIDIA stock data
    df = pd.read_csv(
        DATA_PATH,
        parse_dates=["Date"]
    )

    # Sort by date
    df = df.sort_values("Date")

    # 1. Daily percentage returns
    df["Daily_Return"] = df["Close"].pct_change()

    # 2. Five-day moving average
    df["MA_5"] = df["Close"].rolling(
        window=5
    ).mean()

    # 3. Twenty-day moving average
    df["MA_20"] = df["Close"].rolling(
        window=20
    ).mean()

    # 4. Ten-day rolling volatility
    df["Volatility_10"] = (
        df["Daily_Return"]
        .rolling(window=10)
        .std()
    )

    # Remove rows with incomplete indicators
    df = df.dropna()

    # Save processed dataset
    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(OUTPUT_PATH, index=False)

    print("Feature engineering completed!")
    print(df.tail())

    return df


if __name__ == "__main__":
    create_features()
