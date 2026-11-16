
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
import matplotlib.pyplot as plt


# File locations
DATA_PATH = Path("data/processed/nvda_features.csv")
MODEL_DIR = Path("models")
RESULTS_DIR = Path("results")

FEATURES = [
    "Close",
    "Volume",
    "Daily_Return",
    "MA_5",
    "MA_20",
    "Volatility_10",
]


def train_model():
    # 1. Load processed data
    df = pd.read_csv(
        DATA_PATH,
        parse_dates=["Date"],
    )

    df = df.sort_values("Date").reset_index(drop=True)

    # 2. Create next-trading-day target
    df["Target"] = df["Close"].shift(-1)

    # Remove rows without complete features/target
    data = df.dropna(
        subset=FEATURES + ["Target"]
    ).copy()

    if len(data) < 100:
        raise ValueError("Not enough data for training.")

    # 3. Prepare features and target
    X = data[FEATURES]
    y = data["Target"]

    # 4. Chronological train/test split
    split = int(len(data) * 0.8)

    X_train = X.iloc[:split]
    X_test = X.iloc[split:]

    y_train = y.iloc[:split]
    y_test = y.iloc[split:]

    # 5. Train linear regression
    model = LinearRegression()
    model.fit(X_train, y_train)

    # 6. Predict on test data
    predictions = model.predict(X_test)

    # 7. Evaluate model
    mae = mean_absolute_error(y_test, predictions)

    rmse = np.sqrt(
        mean_squared_error(y_test, predictions)
    )

    r2 = r2_score(y_test, predictions)

    # Naive baseline: tomorrow = today
    baseline_predictions = X_test["Close"].to_numpy()

    baseline_mae = mean_absolute_error(
        y_test,
        baseline_predictions,
    )

    baseline_rmse = np.sqrt(
        mean_squared_error(
            y_test,
            baseline_predictions,
        )
    )

    metrics = {
        "model_mae": float(mae),
        "model_rmse": float(rmse),
        "model_r2": float(r2),
        "baseline_mae": float(baseline_mae),
        "baseline_rmse": float(baseline_rmse),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "train_end_date": str(
            data["Date"].iloc[split - 1].date()
        ),
        "test_start_date": str(
            data["Date"].iloc[split].date()
        ),
    }

    # 8. Create output folders
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "figures").mkdir(
        parents=True,
        exist_ok=True,
    )

    # 9. Save trained model
    joblib.dump(
        model,
        MODEL_DIR / "linear_regression.pkl",
    )

    # Save metrics
    with open(
        RESULTS_DIR / "metrics.json",
        "w",
    ) as file:
        json.dump(metrics, file, indent=4)

    # Save predictions
    results_df = pd.DataFrame({
        "Date": data["Date"].iloc[split:].values,
        "Actual": y_test.to_numpy(),
        "Predicted": predictions,
        "Baseline": baseline_predictions,
    })

    results_df.to_csv(
        RESULTS_DIR / "predictions.csv",
        index=False,
    )

    # 10. Plot actual vs predicted prices
    plt.figure(figsize=(12, 6))

    plt.plot(
        results_df["Date"],
        results_df["Actual"],
        label="Actual Price",
    )

    plt.plot(
        results_df["Date"],
        results_df["Predicted"],
        label="Predicted Price",
        alpha=0.8,
    )

    plt.title(
        "NVIDIA: Actual vs Predicted Closing Price"
    )
    plt.xlabel("Date")
    plt.ylabel("Adjusted Closing Price (USD)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR / "figures" / "predictions.png",
        dpi=150,
    )
    plt.close()

    # 11. Print results
    print("\nMODEL TRAINING COMPLETED")
    print("-" * 35)

    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")

    print(f"\nModel MAE: ${mae:.4f}")
    print(f"Model RMSE: ${rmse:.4f}")
    print(f"Model R²: {r2:.4f}")

    print(f"\nBaseline MAE: ${baseline_mae:.4f}")
    print(f"Baseline RMSE: ${baseline_rmse:.4f}")

    if mae < baseline_mae:
        print("\nModel beats the baseline on MAE.")
    else:
        print("\nModel does not beat the baseline on MAE.")

    print("\nSaved model and evaluation results.")

    return model, metrics


if __name__ == "__main__":
    train_model()
