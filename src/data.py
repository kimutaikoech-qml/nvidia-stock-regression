
import yfinance as yf
import pandas as pd
from pathlib import Path

OUTPUT_PATH = Path(__file__).resolve().parents[1] / "data/raw/nvda.csv"

def download_nvda_data():
    # Download NVIDIA historical prices
    df = yf.download(
        "NVDA",
        start="2020-01-01",
        auto_adjust=True,
        multi_level_index=False,
        progress=False
    )

    if df is None or df.empty:
        raise ValueError("No NVIDIA data downloaded")

    # Ensure chronological ordering
    df = df.sort_index()

    # Check for duplicate dates
    if df.index.has_duplicates:
        raise ValueError("Duplicate dates detected")

    # Save dataset
    output = OUTPUT_PATH
    output.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(output)

    print("NVIDIA data downloaded successfully!")
    print("\nDataset shape:", df.shape)
    print("\nFirst five rows:")
    print(df.head())
    print("\nMissing values:")
    print(df.isna().sum())

    return df


if __name__ == "__main__":
    download_nvda_data()
