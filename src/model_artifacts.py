
import hashlib
import io
import json
import warnings

import joblib
import numpy as np
import pandas as pd
import sklearn
import streamlit as st
from sklearn.exceptions import InconsistentVersionWarning
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.dashboard_data import DataError
from src.model_config import DATA_PATH, FEATURES, METRICS_PATH, MODEL_PATH, PREDICTIONS_PATH

ARTIFACT_PATHS = (MODEL_PATH, METRICS_PATH, PREDICTIONS_PATH, DATA_PATH)


def _signature():
    missing = [str(p.relative_to(DATA_PATH.parents[2])) for p in ARTIFACT_PATHS if not p.is_file()]
    if missing:
        raise DataError("Missing artifacts: " + ", ".join(missing))
    # Do not follow a model/results symlink to an unrelated or uploaded file.
    if any(p.resolve() != p for p in ARTIFACT_PATHS):
        raise DataError("Model artifacts must be regular files at their fixed project paths, without symlinks.")
    return tuple((p.stat().st_mtime_ns, p.stat().st_size) for p in ARTIFACT_PATHS)


def read_features():
    df = pd.read_csv(DATA_PATH, parse_dates=["Date"])
    missing = set(FEATURES + ["Date"]) - set(df)
    if missing or df.empty:
        raise DataError(f"Processed features are empty or missing columns: {sorted(missing)}")
    df['Date'] = pd.to_datetime(df.Date, errors='raise')
    if df.Date.isna().any() or df.Date.duplicated().any():
        raise DataError("Processed dates must be present and unique.")
    df = df.sort_values('Date').reset_index(drop=True)
    for col in FEATURES:
        df[col] = pd.to_numeric(df[col], errors='raise')
    if np.isinf(df[FEATURES].to_numpy(dtype=float)).any():
        raise DataError("Processed features contain infinite values.")
    if (df[['Close', 'MA_5', 'MA_20']] <= 0).any().any():
        raise DataError("Adjusted prices and moving averages must be positive.")
    if (df[['Volume', 'Volatility_10']] < 0).any().any():
        raise DataError("Volume and rolling volatility cannot be negative.")
    return df


def validate_evaluation(metrics, predictions, features, model):
    """Recompute saved errors and check one-step targets against the source CSV."""
    required = {'Date', 'ObservationDate', 'Actual', 'Predicted', 'Baseline'}
    if not required.issubset(predictions) or len(predictions) < 2:
        raise DataError("Predictions need Date, ObservationDate, Actual, Predicted, and Baseline, with at least two rows.")
    p = predictions.copy()
    for col in ['Date', 'ObservationDate']:
        p[col] = pd.to_datetime(p[col], errors='raise')
        if p[col].isna().any() or p[col].duplicated().any() or not p[col].is_monotonic_increasing:
            raise DataError("Prediction dates must be complete, unique, and chronological.")
    for col in ['Actual', 'Predicted', 'Baseline']:
        p[col] = pd.to_numeric(p[col], errors='raise')
    if not np.isfinite(p[['Actual', 'Predicted', 'Baseline']].to_numpy()).all():
        raise DataError("Saved predictions contain missing or invalid values.")

    supervised = features.assign(Target=features.Close.shift(-1),
                                 TargetDate=features.Date.shift(-1)).dropna(subset=FEATURES + ['Target', 'TargetDate'])
    train_end = pd.Timestamp(metrics['train_end_date'])
    test_start = pd.Timestamp(metrics['test_start_date'])
    if train_end >= test_start:
        raise DataError("Training and test observation periods overlap.")
    train = supervised.loc[supervised.Date <= train_end]
    test = supervised.loc[supervised.Date.between(p.ObservationDate.iloc[0], p.ObservationDate.iloc[-1])]
    if (len(train) != metrics['train_samples'] or len(test) != metrics['test_samples']
            or len(p) != len(test) or len(train) < 1):
        raise DataError("Sample counts do not match the training/evaluation artifacts.")
    if (train.Date.iloc[-1] != train_end or test.Date.iloc[0] != test_start
            or train.TargetDate.iloc[-1] >= test.TargetDate.iloc[0]):
        raise DataError("Training and testing target periods are not chronological.")
    if not np.array_equal(test.Date.to_numpy(), p.ObservationDate.to_numpy()):
        raise DataError("Prediction observation dates do not match the processed rows.")
    if (not np.array_equal(test.TargetDate.to_numpy(), p.Date.to_numpy())
            or not (p.Date > p.ObservationDate).all()):
        raise DataError("Prediction dates must be the next observed trading session, not the feature dates.")
    for key, actual in [('train_target_end_date', train.TargetDate.iloc[-1]),
                        ('test_target_start_date', test.TargetDate.iloc[0]),
                        ('test_end_date', test.Date.iloc[-1]),
                        ('test_target_end_date', test.TargetDate.iloc[-1])]:
        if pd.Timestamp(metrics[key]) != actual:
            raise DataError(f"Saved {key} does not match the source rows.")
    for actual, expected, label in [(p.Actual, test.Target, 'actual prices'),
                                    (p.Baseline, test.Close, 'previous-close baseline'),
                                    (p.Predicted, model.predict(test[FEATURES]), 'model predictions')]:
        if not np.allclose(np.asarray(actual), np.asarray(expected), rtol=1e-9, atol=1e-9):
            raise DataError(f"Saved {label} do not match the model/source data.")
    calculated = {
        'model_mae': mean_absolute_error(p.Actual, p.Predicted),
        'model_rmse': np.sqrt(mean_squared_error(p.Actual, p.Predicted)),
        'model_r2': r2_score(p.Actual, p.Predicted),
        'baseline_mae': mean_absolute_error(p.Actual, p.Baseline),
        'baseline_rmse': np.sqrt(mean_squared_error(p.Actual, p.Baseline)),
    }
    for key, actual in calculated.items():
        if not np.isfinite(metrics[key]) or not np.isclose(metrics[key], actual, rtol=1e-9, atol=1e-9):
            raise DataError(f"Saved metric {key} does not match the predictions.")
    p['Residual'] = p.Actual - p.Predicted
    return p


@st.cache_resource(show_spinner=False)
def _load_verified(signature):
    # Cache on all four files. Training is never called from the dashboard.
    try:
        metrics = json.loads(METRICS_PATH.read_text())
        if metrics['artifact_schema_version'] != 2 or metrics['feature_names'] != FEATURES:
            raise DataError("Training metadata has an incompatible feature contract. Regenerate artifacts with src/train.py.")
        if metrics['target'] != 'next_trading_session_adjusted_close':
            raise DataError("The saved model has an incompatible prediction target.")
        if metrics['sklearn_version'] != sklearn.__version__:
            raise DataError(f"Model requires scikit-learn {metrics['sklearn_version']}; installed {sklearn.__version__}.")
        model_bytes = MODEL_PATH.read_bytes()
        prediction_bytes = PREDICTIONS_PATH.read_bytes()
        for payload, key in [(model_bytes, 'model_sha256'), (prediction_bytes, 'predictions_sha256')]:
            if hashlib.sha256(payload).hexdigest() != metrics[key]:
                raise DataError("Model/result integrity check failed. Regenerate a consistent set of local artifacts.")
        with warnings.catch_warnings():
            warnings.simplefilter('error', InconsistentVersionWarning)
            model = joblib.load(io.BytesIO(model_bytes))
        if type(model) is not LinearRegression or list(model.feature_names_in_) != FEATURES:
            raise DataError("Saved model is not the expected fitted Multiple Linear Regression model/feature order.")
        if (not np.isfinite(model.coef_).all() or not np.isfinite(model.intercept_)
                or model.coef_.shape != (len(FEATURES),)):
            raise DataError("Saved model coefficients are invalid.")
        features = read_features()
        predictions = pd.read_csv(io.BytesIO(prediction_bytes))
        predictions = validate_evaluation(metrics, predictions, features, model)
        return {'model': model, 'metrics': metrics, 'predictions': predictions, 'features': features}
    except DataError:
        raise
    except Exception as exc:
        raise DataError(f"Could not validate model artifacts: {exc}") from exc


def load_artifacts():
    try:
        return _load_verified(_signature())
    except OSError as exc:
        raise DataError(f"Cannot read model artifacts: {exc}") from exc


def forecast_next_session(model, features):
    # Preserve the fitted feature order even if the CSV columns are rearranged.
    names = list(model.feature_names_in_)
    if names != FEATURES:
        raise DataError("Forecast feature order does not match training.")
    complete = features.dropna(subset=names).sort_values('Date')
    if complete.empty:
        raise DataError("No complete feature row is available for forecasting.")
    latest = complete.iloc[[-1]]
    inputs = latest.loc[:, names]
    if not np.isfinite(inputs.to_numpy(dtype=float)).all():
        raise DataError("Latest feature row has invalid values.")
    estimate = float(model.predict(inputs)[0])
    close = float(latest.Close.iloc[0])
    if not np.isfinite(estimate) or not np.isfinite(close) or close <= 0:
        raise DataError("The forecast or reference close is invalid.")
    return {'observation_date': latest.Date.iloc[0], 'latest_close': close,
            'predicted_close': estimate, 'dollar_change': estimate - close,
            'percent_change': (estimate / close - 1) * 100,
            'dataset_latest_date': features.Date.max()}
