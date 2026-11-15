
import json
import hashlib
import platform

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
import matplotlib.pyplot as plt
import sklearn

if __package__:
    from .model_config import (ARTIFACT_SCHEMA_VERSION, DATA_PATH, EVALUATION_NOTE,
                               FEATURES, METRICS_PATH, MODEL_PATH, PREDICTIONS_PATH)
    from .regression import build_model, coefficient_table, original_intercept
else:
    from model_config import (ARTIFACT_SCHEMA_VERSION, DATA_PATH, EVALUATION_NOTE,
                              FEATURES, METRICS_PATH, MODEL_PATH, PREDICTIONS_PATH)
    from regression import build_model, coefficient_table, original_intercept


# File locations
MODEL_DIR = MODEL_PATH.parent
RESULTS_DIR = METRICS_PATH.parent


def train_model():
    # 1. Load processed data
    df = pd.read_csv(
        DATA_PATH,
        parse_dates=["Date"],
    )

    df = df.sort_values("Date").reset_index(drop=True)
    if df["Date"].isna().any() or df["Date"].duplicated().any():
        raise ValueError("Observation dates must be present and unique.")
    for column in FEATURES:
        df[column] = pd.to_numeric(df[column], errors="raise")
    if np.isinf(df[FEATURES].to_numpy(dtype=float)).any():
        raise ValueError("Features must not contain infinite values.")

    # 2. Create next-trading-day target
    df["Target"] = df["Close"].shift(-1)
    df["TargetDate"] = df["Date"].shift(-1)

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
    model = build_model()
    model.fit(X_train, y_train)
    regression = model.named_steps['regression']
    coefficients = coefficient_table(model)

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
        "train_target_end_date": str(data["TargetDate"].iloc[split - 1].date()),
        "test_target_start_date": str(data["TargetDate"].iloc[split].date()),
        "test_end_date": str(data["Date"].iloc[-1].date()),
        "test_target_end_date": str(data["TargetDate"].iloc[-1].date()),
        "feature_names": FEATURES,
        "sklearn_version": sklearn.__version__,
        "joblib_version": joblib.__version__,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "python_version": platform.python_version(),
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "target": "next_trading_session_adjusted_close",
        "preprocessing": "StandardScaler fitted on training data only",
        "effective_rank": int(regression.rank_),
        "singular_values": regression.singular_.tolist(),
        "coefficients_standardized": dict(zip(FEATURES, regression.coef_.tolist())),
        "coefficients_original_units": dict(zip(FEATURES, coefficients['Original-unit coefficient'].tolist())),
        "intercept_standardized": float(regression.intercept_),
        "intercept_original_units": original_intercept(model),
        "scaler_fit_samples": int(model.named_steps['scaler'].n_samples_seen_),
        "test_previously_inspected": True,
        "evaluation_note": EVALUATION_NOTE,
        "processed_data_sha256": hashlib.sha256(DATA_PATH.read_bytes()).hexdigest(),
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
        MODEL_PATH,
    )

    # Save predictions
    results_df = pd.DataFrame({
        # Chart prices at their target trading date, not the input date.
        "Date": data["TargetDate"].iloc[split:].values,
        "ObservationDate": data["Date"].iloc[split:].values,
        "Actual": y_test.to_numpy(),
        "Predicted": predictions,
        "Baseline": baseline_predictions,
    })

    results_df.to_csv(
        PREDICTIONS_PATH,
        index=False,
    )
    # These hashes detect stale/mixed artifacts; they are not proof of trust.
    metrics["model_sha256"] = hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest()
    metrics["predictions_sha256"] = hashlib.sha256(PREDICTIONS_PATH.read_bytes()).hexdigest()
    with METRICS_PATH.open("w") as file:
        json.dump(metrics, file, indent=4, allow_nan=False)

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
    print(f"Effective numerical rank: {regression.rank_}/{len(FEATURES)}")
    print("\nCoefficients (USD per training standard deviation / original feature unit):")
    print(coefficients.to_string(index=False))
    print(f"Standardized intercept: {regression.intercept_:.8f}")
    print(f"Original-unit intercept: {original_intercept(model):.8f}")

    print(f"\nBaseline MAE: ${baseline_mae:.4f}")
    print(f"Baseline RMSE: ${baseline_rmse:.4f}")

    if mae < baseline_mae:
        print("\nModel beats the baseline on MAE.")
    else:
        print("\nModel does not beat the baseline on MAE.")
    print("\n" + EVALUATION_NOTE)

    print("\nSaved model and evaluation results.")

    return model, metrics


if __name__ == "__main__":
    train_model()
