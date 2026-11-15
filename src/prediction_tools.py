"""Read-only input preparation for interactive forecasts and saved test results."""
import hashlib

import numpy as np
import pandas as pd

from src.dashboard_data import DataError
from src.model_config import FEATURES


def interaction_token(bundle):
    """Invalidate session results when the model, evaluation, or feature data changes."""
    payload = pd.util.hash_pandas_object(bundle['features'][['Date'] + FEATURES], index=False).values.tobytes()
    return (bundle['metrics']['model_sha256'], bundle['metrics']['predictions_sha256'],
            hashlib.sha256(payload).hexdigest())


def latest_complete_row(features):
    missing = set(['Date'] + FEATURES) - set(features)
    if missing:
        raise DataError(f"Missing forecast inputs: {', '.join(sorted(missing))}")
    try:
        df = features[['Date'] + FEATURES].copy()
        df['Date'] = pd.to_datetime(df.Date, errors='raise')
        for name in FEATURES:
            df[name] = pd.to_numeric(df[name], errors='raise')
        complete = df.dropna(subset=['Date'] + FEATURES).sort_values('Date')
        if complete.empty:
            raise DataError('No complete feature row is available for forecasting.')
        latest = complete.tail(1)
        if not np.isfinite(latest[FEATURES].to_numpy(dtype=float)).all():
            raise DataError('Latest feature row contains non-finite values.')
        return latest
    except (ValueError, TypeError) as exc:
        raise DataError(f'Cannot prepare forecast inputs: {exc}') from exc


def training_ranges(features, metrics):
    train = features.loc[features.Date <= pd.Timestamp(metrics['train_end_date']), FEATURES].dropna()
    if train.empty or not np.isfinite(train.to_numpy(dtype=float)).all():
        raise DataError('Valid historical training ranges are unavailable.')
    return train.agg(['min', 'max'])


def simulate(model, values, ranges):
    """Predict a hypothetical row without fitting or mutating the saved pipeline."""
    if list(model.feature_names_in_) != FEATURES:
        raise DataError('Simulator feature order does not match the trained pipeline.')
    if set(values) != set(FEATURES):
        raise DataError('Provide all six model inputs with the expected feature names.')
    try:
        # Reject booleans, numeric-looking strings, NaN, and infinities explicitly.
        if any(isinstance(values[name], (bool, str)) or not np.isscalar(values[name]) for name in FEATURES):
            raise ValueError('Inputs must be numeric values.')
        frame = pd.DataFrame([[float(values[name]) for name in FEATURES]], columns=FEATURES)
    except (ValueError, TypeError, OverflowError) as exc:
        raise DataError('All simulator inputs must be numeric and finite.') from exc
    if not np.isfinite(frame.to_numpy()).all():
        raise DataError('All simulator inputs must be numeric and finite.')
    if (frame[['Close', 'MA_5', 'MA_20']] <= 0).any().any():
        raise DataError('Closing price and moving averages must be positive.')
    if values['Volume'] < 0 or values['Volatility_10'] < 0 or values['Daily_Return'] <= -1:
        raise DataError('Volume and volatility must be non-negative; daily return must exceed -100%.')
    outside = [name for name in FEATURES if not ranges.loc['min', name] <= frame[name].iloc[0] <= ranges.loc['max', name]]
    estimate = float(model.predict(frame)[0])
    if not np.isfinite(estimate):
        raise DataError('The simulated prediction is not finite.')
    close = float(values['Close'])
    return {'predicted_close': estimate, 'dollar_change': estimate - close,
            'percent_change': (estimate / close - 1) * 100, 'outside_ranges': outside,
            'inputs': {name: float(values[name]) for name in FEATURES}}


def historical_result(predictions, target_date):
    """Select only an actual saved test prediction, never synthesize a backtest."""
    if not {'Date', 'ObservationDate', 'Actual', 'Predicted'}.issubset(predictions):
        raise DataError('Saved historical prediction columns are missing.')
    try:
        date = pd.Timestamp(target_date)
    except (ValueError, TypeError) as exc:
        raise DataError('Choose a saved target trading date.') from exc
    selected = predictions.loc[pd.to_datetime(predictions.Date) == date]
    if len(selected) != 1:
        raise DataError('No unique saved prediction exists for that trading date.')
    row = selected.iloc[0]
    if pd.Timestamp(row.ObservationDate) >= pd.Timestamp(row.Date):
        raise DataError('The observation date must precede the target trading date.')
    if not np.isfinite([row.Actual, row.Predicted]).all():
        raise DataError('The saved prediction has invalid prices.')
    return row
