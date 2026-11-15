"""Verify model inference, artifacts, target alignment, and failure paths."""
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import subprocess
import sys
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from streamlit.testing.v1 import AppTest

from src.dashboard_data import DataError
from src.dashboard_data import RAW_PATH
from src.model_artifacts import (_load_verified, _signature, forecast_next_session,
                                 load_artifacts, validate_evaluation)
from src.model_config import FEATURES, METRICS_PATH, MODEL_PATH, ROOT
from src.regression import build_model, coefficient_table, original_intercept


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = load_artifacts()

    def validate(self, metrics=None, predictions=None):
        b = self.bundle
        return validate_evaluation(metrics if metrics is not None else b['metrics'],
                                   predictions if predictions is not None else b['predictions'],
                                   b['features'], b['model'])

    def test_metrics_and_dates_recompute(self):
        p = self.validate()
        self.assertTrue((p.Date > p.ObservationDate).all())
        self.assertEqual(len(p), self.bundle['metrics']['test_samples'])
        np.testing.assert_allclose(p.Residual, p.Actual - p.Predicted)

    def test_saved_pipeline_has_full_rank_and_training_only_scaler(self):
        b = self.bundle
        model = b['model']
        self.assertIsInstance(model, Pipeline)
        self.assertEqual(list(model.named_steps), ['scaler', 'regression'])
        self.assertIsInstance(model.named_steps['scaler'], StandardScaler)
        train = b['features'].loc[b['features'].Date <= pd.Timestamp(b['metrics']['train_end_date']), FEATURES]
        scaler = model.named_steps['scaler']
        self.assertEqual(scaler.n_samples_seen_, len(train))
        np.testing.assert_allclose(scaler.mean_, train.mean().to_numpy())
        np.testing.assert_allclose(scaler.var_, train.var(ddof=0).to_numpy())
        self.assertFalse(np.allclose(scaler.mean_, b['features'][FEATURES].mean().to_numpy()))
        self.assertEqual(model.named_steps['regression'].rank_, len(FEATURES))
        self.assertEqual(b['metrics']['effective_rank'], len(FEATURES))
        self.assertTrue(b['metrics']['test_previously_inspected'])

    def test_pipeline_does_not_fit_scaler_when_predicting(self):
        features = self.bundle['features'][FEATURES]
        model = build_model()
        model.fit(features.iloc[:100], self.bundle['features'].Close.iloc[1:101])
        mean_before = model.named_steps['scaler'].mean_.copy()
        with patch('sklearn.preprocessing.StandardScaler.fit', side_effect=AssertionError('Unexpected scaler fit')):
            model.predict(features.tail(2))
        np.testing.assert_array_equal(model.named_steps['scaler'].mean_, mean_before)

    def test_coefficients_in_both_units_match_pipeline_predictions(self):
        model = self.bundle['model']
        inputs = self.bundle['features'][FEATURES].tail(10)
        table = coefficient_table(model)
        expected = model.predict(inputs)
        original = inputs.to_numpy() @ table['Original-unit coefficient'].to_numpy() + original_intercept(model)
        np.testing.assert_allclose(expected, original, rtol=1e-10, atol=1e-10)
        standardized = model.named_steps['scaler'].transform(inputs) @ table['Standardized coefficient'].to_numpy()
        standardized += model.named_steps['regression'].intercept_
        np.testing.assert_allclose(expected, standardized)

    def test_complete_pipeline_serialization_and_input_contract(self):
        model = self.bundle['model']
        inputs = self.bundle['features'][FEATURES].tail(2)
        with TemporaryDirectory() as folder:
            path = Path(folder) / 'trusted-test-model.pkl'
            joblib.dump(model, path)
            restored = joblib.load(path)  # Only our locally generated test artifact.
            np.testing.assert_allclose(model.predict(inputs), restored.predict(inputs))
        with self.assertRaises(ValueError):
            model.predict(inputs[list(reversed(FEATURES))])

    def test_scaler_fitted_to_test_data_is_rejected(self):
        b = self.bundle
        model = build_model()
        model.fit(b['features'][FEATURES], b['features'].Close)
        with self.assertRaisesRegex(DataError, 'Scaler statistics'):
            validate_evaluation(b['metrics'], b['predictions'], b['features'], model)

    def test_scripts_resolve_paths_outside_project(self):
        with TemporaryDirectory() as folder:
            code = (
                'import runpy; from pathlib import Path; '
                f'root=Path({str(ROOT)!r}); '
                "data=runpy.run_path(str(root/'src/data.py')); "
                "features=runpy.run_path(str(root/'src/features.py')); "
                "assert data['OUTPUT_PATH']==root/'data/raw/nvda.csv'; "
                "assert features['DATA_PATH']==root/'data/raw/nvda.csv'; "
                "assert features['OUTPUT_PATH']==root/'data/processed/nvda_features.csv'"
            )
            subprocess.run([sys.executable, '-c', code], cwd=folder, check=True, capture_output=True)

    def test_dashboard_metrics_match_corrected_artifacts(self):
        b = self.bundle
        app = AppTest.from_file(str(ROOT / 'app.py'), default_timeout=30).run()
        app.sidebar.radio[0].set_value('ML Predictions').run()
        self.assertFalse(app.exception)
        metrics = {m.label: m.value for m in app.metric}
        self.assertEqual(metrics['Model MAE · USD'], f"${b['metrics']['model_mae']:.4f}")
        self.assertEqual(metrics['Model RMSE · USD'], f"${b['metrics']['model_rmse']:.4f}")
        self.assertEqual(metrics['Model R²'], f"{b['metrics']['model_r2']:.4f}")
        forecast = forecast_next_session(b['model'], b['features'])
        self.assertEqual(metrics['Estimated next-session close'], f"${forecast['predicted_close']:,.2f}")
        self.assertTrue(any('previously inspected' in i.value and 'similarly' in i.value
                            and 'previous-close baseline' in i.value for i in app.info))
        self.assertFalse(app.success)  # Trained status is a small caption.
        self.assertFalse(any('StandardScaler' in c.value or 'verified' in c.value for c in app.caption))
        self.assertNotIn('Effective numerical rank', metrics)
        self.assertFalse(app.expander)
        app.sidebar.radio[0].set_value('Model Insights').run()
        self.assertFalse(app.exception)
        self.assertTrue(any('StandardScaler' in c.value for c in app.caption))
        self.assertTrue(any(m.label == 'Effective numerical rank' for m in app.metric))
        self.assertTrue(any(e.label == 'Forecast inputs and timing' for e in app.expander))

    def test_saved_indicators_use_only_current_and_past_prices(self):
        raw = pd.read_csv(RAW_PATH, parse_dates=['Date']).sort_values('Date')
        expected = raw[['Date']].copy()
        expected['Daily_Return'] = raw.Close.pct_change()
        expected['MA_5'] = raw.Close.rolling(5).mean()
        expected['MA_20'] = raw.Close.rolling(20).mean()
        expected['Volatility_10'] = expected.Daily_Return.rolling(10).std()
        joined = self.bundle['features'].merge(expected, on='Date', suffixes=('', '_expected'))
        for name in ['Daily_Return', 'MA_5', 'MA_20', 'Volatility_10']:
            np.testing.assert_allclose(joined[name], joined[name + '_expected'], rtol=1e-9, atol=1e-9)

    def test_bad_target_dates_rejected(self):
        p = self.bundle['predictions'].copy()
        p['Date'] = p.ObservationDate
        with self.assertRaisesRegex(DataError, 'next observed trading session'):
            self.validate(predictions=p)

    def test_future_target_as_baseline_rejected(self):
        p = self.bundle['predictions'].copy()
        p['Baseline'] = p.Actual
        with self.assertRaisesRegex(DataError, 'previous-close baseline'):
            self.validate(predictions=p)

    def test_bad_metrics_and_overlap_rejected(self):
        m = dict(self.bundle['metrics'])
        m['model_mae'] += 1
        with self.assertRaisesRegex(DataError, 'metric model_mae'):
            self.validate(metrics=m)
        m = dict(self.bundle['metrics'])
        m['train_end_date'] = m['test_start_date']
        with self.assertRaisesRegex(DataError, 'overlap'):
            self.validate(metrics=m)

    def test_invalid_predictions_rejected(self):
        p = self.bundle['predictions'].copy()
        p.loc[p.index[0], 'Predicted'] = np.nan
        with self.assertRaisesRegex(DataError, 'invalid values'):
            self.validate(predictions=p)

    def test_forecast_uses_exact_feature_order(self):
        b = self.bundle
        reordered = b['features'][list(reversed(b['features'].columns))]
        result = forecast_next_session(b['model'], reordered)
        expected = b['model'].predict(b['features'].tail(1)[FEATURES])[0]
        self.assertEqual(result['predicted_close'], expected)
        self.assertEqual(result['observation_date'], b['features'].Date.max())
        self.assertAlmostEqual(result['dollar_change'], expected - result['latest_close'])
        self.assertAlmostEqual(result['percent_change'], result['dollar_change'] / result['latest_close'] * 100)
        self.assertNotIn('forecast_date', result)  # No calendar-day guess.

    def test_latest_complete_row_and_no_complete_row(self):
        b = self.bundle
        incomplete = b['features'].copy()
        incomplete.loc[incomplete.index[-1], 'Volume'] = np.nan
        result = forecast_next_session(b['model'], incomplete)
        self.assertEqual(result['observation_date'], incomplete.Date.iloc[-2])
        incomplete['Volume'] = np.nan
        with self.assertRaisesRegex(DataError, 'No complete feature row'):
            forecast_next_session(b['model'], incomplete)

    def test_version_and_hash_rejected_before_deserialization(self):
        with TemporaryDirectory() as folder:
            path = Path(folder) / 'metrics.json'
            for key, value in [('sklearn_version', '0.0'), ('model_sha256', 'invalid')]:
                metadata = dict(self.bundle['metrics'])
                metadata[key] = value
                path.write_text(json.dumps(metadata))
                with patch('src.model_artifacts.METRICS_PATH', path), patch('src.model_artifacts.joblib.load') as load:
                    _load_verified.clear()
                    with self.assertRaises(DataError):
                        _load_verified(('incompatible', key))
                    load.assert_not_called()
        _load_verified.clear()

    def test_missing_and_symlink_artifacts_rejected(self):
        with patch('src.model_artifacts.ARTIFACT_PATHS', (MODEL_PATH.with_name('nonexistent.pkl'),)):
            with self.assertRaisesRegex(DataError, 'Missing artifacts'):
                _signature()
        with TemporaryDirectory() as folder:
            target = Path(folder) / 'model.pkl'
            target.write_bytes(b'untrusted')
            link = Path(folder) / 'link.pkl'
            link.symlink_to(target)
            with patch('src.model_artifacts.ARTIFACT_PATHS', (link,)):
                with self.assertRaisesRegex(DataError, 'without symlinks'):
                    _signature()

    def test_missing_artifacts_both_model_pages(self):
        with patch('src.model_pages.load_artifacts', side_effect=DataError('Missing model artifacts')):
            app = AppTest.from_file(str(ROOT / 'app.py'), default_timeout=30).run()
            for page in ['ML Predictions', 'Model Insights']:
                app.sidebar.radio[0].set_value(page).run()
                self.assertFalse(app.exception)
                self.assertTrue(any('Model results are unavailable' in w.value for w in app.warning))
                self.assertFalse(any('Missing model artifacts' in w.value for w in app.warning))
                self.assertTrue(app.info)
                self.assertFalse(app.get('plotly_chart'))

    def test_reruns_never_train(self):
        with patch('sklearn.linear_model.LinearRegression.fit', side_effect=AssertionError('Unexpected training')), \
                patch('sklearn.preprocessing.StandardScaler.fit', side_effect=AssertionError('Unexpected preprocessing fit')):
            app = AppTest.from_file(str(ROOT / 'app.py'), default_timeout=30).run()
            for page in ['ML Predictions', 'Model Insights', 'ML Predictions']:
                app.sidebar.radio[0].set_value(page).run()
                self.assertFalse(app.exception)
                self.assertFalse(app.error)


if __name__ == '__main__':
    unittest.main()
