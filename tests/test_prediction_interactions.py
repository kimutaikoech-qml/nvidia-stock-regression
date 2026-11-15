"""Interactive ML checks using the existing pipeline; never regenerate artifacts."""
import json
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

from src.dashboard_data import DataError
from src.model_artifacts import forecast_next_session, load_artifacts
from src.model_config import FEATURES, ROOT
from src.prediction_tools import (historical_result, latest_complete_row, simulate,
                                  training_ranges)


class InputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = load_artifacts()
        cls.latest = latest_complete_row(cls.bundle['features'])
        cls.values = cls.latest.iloc[0][FEATURES].to_dict()
        cls.ranges = training_ranges(cls.bundle['features'], cls.bundle['metrics'])

    def test_simulation_feature_order_and_pipeline_output(self):
        values = dict(reversed(list(self.values.items())))
        result = simulate(self.bundle['model'], values, self.ranges)
        expected = self.bundle['model'].predict(pd.DataFrame([self.values], columns=FEATURES))[0]
        self.assertAlmostEqual(result['predicted_close'], expected)
        self.assertEqual(list(result['inputs']), FEATURES)
        self.assertAlmostEqual(result['dollar_change'], expected - values['Close'])
        self.assertAlmostEqual(result['percent_change'], (expected / values['Close'] - 1) * 100)

    def test_invalid_and_missing_simulator_inputs(self):
        for bad in [None, '123', True, np.nan, np.inf, -np.inf]:
            with self.subTest(bad=bad):
                values = dict(self.values, Close=bad)
                with self.assertRaises(DataError):
                    simulate(self.bundle['model'], values, self.ranges)
        for name, value in [('Close', 0), ('MA_5', -1), ('Volume', -1), ('Daily_Return', -1), ('Volatility_10', -1)]:
            with self.subTest(name=name):
                with self.assertRaises(DataError):
                    simulate(self.bundle['model'], dict(self.values, **{name: value}), self.ranges)
        missing = dict(self.values)
        del missing['Volume']
        with self.assertRaises(DataError):
            simulate(self.bundle['model'], missing, self.ranges)

    def test_training_ranges_exclude_test_period(self):
        train = self.bundle['features'].loc[self.bundle['features'].Date <= self.bundle['metrics']['train_end_date']]
        for name in FEATURES:
            self.assertEqual(self.ranges.loc['min', name], train[name].min())
            self.assertEqual(self.ranges.loc['max', name], train[name].max())
        result = simulate(self.bundle['model'], dict(self.values, Close=self.ranges.loc['max', 'Close'] * 2), self.ranges)
        self.assertIn('Close', result['outside_ranges'])

    def test_latest_complete_features_and_invalid_rows(self):
        df = self.bundle['features'].copy()
        df.loc[df.index[-1], 'Volume'] = np.nan
        self.assertEqual(latest_complete_row(df).Date.iloc[0], df.Date.iloc[-2])
        with self.assertRaises(DataError):
            latest_complete_row(df.drop(columns='MA_5'))
        df['Close'] = np.nan
        with self.assertRaises(DataError):
            latest_complete_row(df)
        df = self.bundle['features'].copy()
        df['Volume'] = df.Volume.astype(float)
        df.loc[df.index[-1], 'Volume'] = np.inf
        with self.assertRaises(DataError):
            latest_complete_row(df)

    def test_historical_selection_uses_saved_target_date(self):
        p = self.bundle['predictions']
        selected = historical_result(p, p.Date.iloc[0].date())
        self.assertEqual(selected.Predicted, p.Predicted.iloc[0])
        self.assertGreater(selected.Date, selected.ObservationDate)
        with self.assertRaises(DataError):
            historical_result(p, '2020-01-01')
        bad = p.copy()
        bad['ObservationDate'] = bad.Date
        with self.assertRaises(DataError):
            historical_result(bad, bad.Date.iloc[0])


class InteractiveDashboardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = load_artifacts()

    def app(self):
        app = AppTest.from_file(str(ROOT / 'app.py'), default_timeout=30).run()
        app.sidebar.radio[0].set_value('ML Predictions').run()
        self.assertFalse(app.exception)
        return app

    def click(self, app, label):
        next(button for button in app.button if button.label == label).click().run()
        self.assertFalse(app.exception, [e.message for e in app.exception])

    def test_prediction_only_on_click_and_persists_during_exploration(self):
        with patch('src.prediction_widgets.forecast_next_session', wraps=forecast_next_session) as generate:
            app = self.app()
            self.assertEqual(generate.call_count, 0)
            self.assertFalse(any(m.label == 'Estimated next-session close' for m in app.metric))
            self.click(app, 'Generate Prediction')
            self.assertEqual(generate.call_count, 1)
            expected = forecast_next_session(self.bundle['model'], self.bundle['features'])
            metric = next(m for m in app.metric if m.label == 'Estimated next-session close')
            self.assertEqual(metric.value, f"${expected['predicted_close']:,.2f}")
            app.selectbox[0].set_value(self.bundle['predictions'].Date.iloc[0].date()).run()
            self.assertFalse(app.exception)
            self.assertEqual(generate.call_count, 1)
            self.assertTrue(any(m.label == 'Estimated next-session close' for m in app.metric))

    def test_historical_dates_and_chart_highlight(self):
        app = self.app()
        p = self.bundle['predictions']
        self.assertEqual(len(app.selectbox[0].options), len(p))
        app.selectbox[0].set_value(p.Date.iloc[0].date()).run()
        self.assertFalse(app.exception)
        values = {m.label: m.value for m in app.metric}
        self.assertEqual(values['Historical predicted close'], f'${p.Predicted.iloc[0]:,.2f}')
        self.assertEqual(values['Historical actual close'], f'${p.Actual.iloc[0]:,.2f}')
        self.assertEqual(values['Prediction error · predicted − actual'], f'${p.Predicted.iloc[0] - p.Actual.iloc[0]:+,.2f}')
        spec = json.loads(app.get('plotly_chart')[0].proto.spec)
        self.assertEqual(spec['data'][-1]['name'], 'Selected target')
        self.assertEqual(pd.Timestamp(spec['data'][-1]['x'][0]), p.Date.iloc[0])
        self.assertTrue(any('Feature observation:' in c.value and 'target trading session:' in c.value for c in app.caption))

    def test_simulation_units_ranges_and_no_training(self):
        app = self.app()
        original_mean = self.bundle['model'].named_steps['scaler'].mean_.copy()
        original_coefficients = self.bundle['model'].named_steps['regression'].coef_.copy()
        latest = self.bundle['features'].iloc[-1]
        daily = next(n for n in app.number_input if n.key == 'sim_input_Daily_Return')
        self.assertAlmostEqual(daily.value, latest.Daily_Return * 100)
        with patch('sklearn.linear_model.LinearRegression.fit', side_effect=AssertionError('Unexpected training')), \
                patch('sklearn.preprocessing.StandardScaler.fit', side_effect=AssertionError('Unexpected scaler fit')):
            self.click(app, 'Generate Prediction')
            next(n for n in app.number_input if n.key == 'sim_input_Close').set_value(500.)
            self.click(app, 'Run Simulation')
            self.assertTrue(any('Outside historical training ranges' in w.value for w in app.warning))
            inputs = latest[FEATURES].to_dict()
            inputs['Close'] = 500.
            expected = self.bundle['model'].predict(pd.DataFrame([inputs], columns=FEATURES))[0]
            result = next(m for m in app.metric if m.label == 'Hypothetical next-session close')
            self.assertEqual(result.value, f'${expected:,.2f}')
            app.run()
            self.assertFalse(app.exception)
            self.assertTrue(any(m.label == 'Hypothetical next-session close' for m in app.metric))
        np.testing.assert_array_equal(self.bundle['model'].named_steps['scaler'].mean_, original_mean)
        np.testing.assert_array_equal(self.bundle['model'].named_steps['regression'].coef_, original_coefficients)

    def test_invalid_simulation_is_handled_without_stale_output(self):
        app = self.app()
        self.click(app, 'Run Simulation')
        with patch('src.prediction_widgets.simulate', side_effect=DataError('Invalid numeric input')):
            self.click(app, 'Run Simulation')
            self.assertTrue(any('Invalid numeric input' in w.value for w in app.warning))
            self.assertFalse(any(m.label == 'Hypothetical next-session close' for m in app.metric))

    def test_model_or_data_change_clears_previous_estimates(self):
        app = self.app()
        self.click(app, 'Generate Prediction')
        self.click(app, 'Run Simulation')
        changed = dict(self.bundle, features=self.bundle['features'].copy())
        changed['features'].loc[changed['features'].index[-1], 'Close'] += .1
        with patch('src.model_pages.load_artifacts', return_value=changed):
            app.run()
            self.assertFalse(app.exception)
            self.assertFalse(any(m.label in ['Estimated next-session close', 'Hypothetical next-session close'] for m in app.metric))

    def test_missing_features_leave_evaluation_available(self):
        changed = dict(self.bundle, features=self.bundle['features'].drop(columns='MA_5'))
        with patch('src.model_pages.load_artifacts', return_value=changed):
            app = self.app()
            self.assertTrue(any('Interactive inputs are unavailable' in w.value for w in app.warning))
            self.assertFalse(any(b.label == 'Generate Prediction' for b in app.button))
            self.assertTrue(any(m.label == 'Model MAE · USD' for m in app.metric))


if __name__ == '__main__':
    unittest.main()
