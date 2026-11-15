"""Read-only dashboard checks: python -m unittest discover -s tests -v."""
from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import pandas as pd
from streamlit.testing.v1 import AppTest

from src.dashboard_data import (DataError, FEATURES_PATH, RAW_PATH, check_feature_prices,
                                filter_dates, load_dataset, price_column)

APP = Path(__file__).resolve().parents[1] / "app.py"


class DataTests(unittest.TestCase):
    def test_saved_sources_agree(self):
        raw = load_dataset(RAW_PATH)
        features = load_dataset(FEATURES_PATH, processed=True)
        check_feature_prices(raw, features)
        self.assertTrue(raw.Date.is_monotonic_increasing)
        self.assertTrue(features.Date.is_monotonic_increasing)

    def test_inclusive_filter_and_empty_range(self):
        df = pd.DataFrame({"Date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-05"])})
        selected = filter_dates(df, date(2024, 1, 2), date(2024, 1, 3))
        self.assertEqual(len(selected), 2)
        self.assertTrue(filter_dates(df, date(2024, 1, 6), date(2024, 1, 7)).empty)

    def test_missing_and_malformed_files(self):
        with TemporaryDirectory() as folder:
            path = Path(folder) / "bad.csv"
            with self.assertRaises(DataError):
                load_dataset(path)
            for text in ["", "Date,Close\nbad,abc\n"]:
                path.write_text(text)
                with self.assertRaises(DataError):
                    load_dataset(path)

    def test_cache_follows_csv_changes(self):
        with TemporaryDirectory() as folder:
            path = Path(folder) / "test.csv"
            df = load_dataset(RAW_PATH).head(2).copy()
            df.to_csv(path, index=False)
            self.assertEqual(len(load_dataset(path)), 2)
            df.head(1).to_csv(path, index=False)
            self.assertEqual(len(load_dataset(path)), 1)

    def test_explicit_adjusted_price_and_mismatch(self):
        raw = load_dataset(RAW_PATH).head(2).copy()
        raw['Adj Close'] = raw.Close * .5
        self.assertEqual(price_column(raw), 'Adj Close')
        with self.assertRaises(DataError):
            check_feature_prices(raw, raw.drop(columns=['Adj Close']))


class DashboardTests(unittest.TestCase):
    def app(self):
        return AppTest.from_file(str(APP), default_timeout=30).run()

    def assert_clean(self, app):
        self.assertFalse(app.exception, [e.message for e in app.exception])
        self.assertFalse(app.error, [e.value for e in app.error])

    def test_all_pages_and_chart_specs(self):
        app = self.app()
        self.assert_clean(app)
        raw = load_dataset(RAW_PATH)
        self.assertEqual(app.metric[0].value, f"${raw[price_column(raw)].iloc[-1]:,.2f}")
        self.assertEqual(app.metric[3].value, f"{len(raw):,}")
        self.assertEqual(len(app.dataframe[0].value), min(20, len(raw)))
        for page, count in [('Overview', 2), ('Market Analysis', 4),
                            ('ML Predictions', 1), ('Model Insights', 4)]:
            app.sidebar.radio[0].set_value(page).run()
            self.assert_clean(app)
            charts = app.get('plotly_chart')
            self.assertEqual(len(charts), count)
            for chart in charts:
                self.assertTrue(chart.proto.spec)
            if page == 'ML Predictions':
                self.assertTrue(all(m.value != '—' for m in app.metric))

    def test_dates_update_table_and_charts(self):
        app = self.app()
        raw = load_dataset(RAW_PATH)
        start, end = raw.Date.iloc[-5].date(), raw.Date.iloc[-1].date()
        app.sidebar.date_input[0].set_value((start, end)).run()
        self.assert_clean(app)
        self.assertEqual(len(app.dataframe[0].value), 5)
        self.assertEqual(len(app.get('plotly_chart')), 2)
        app.sidebar.radio[0].set_value('Market Analysis').run()
        self.assert_clean(app)
        self.assertEqual(len(app.get('plotly_chart')), 4)

    def test_partial_range_and_weekend(self):
        app = self.app()
        app.sidebar.date_input[0].set_value((date(2024, 1, 6),)).run()
        self.assert_clean(app)
        self.assertTrue(app.info)
        app.sidebar.date_input[0].set_value((date(2024, 1, 6), date(2024, 1, 7))).run()
        self.assert_clean(app)
        self.assertFalse(app.get('plotly_chart'))
        self.assertTrue(app.info)

    def test_missing_raw_data_graceful_and_model_pages_accessible(self):
        with patch('src.dashboard_data.load_dataset', side_effect=DataError('Missing CSV')):
            app = self.app()
            self.assertFalse(app.exception)
            self.assertTrue(app.error)
            app.sidebar.radio[0].set_value('ML Predictions').run()
            self.assert_clean(app)
            app.sidebar.radio[0].set_value('Model Insights').run()
            self.assert_clean(app)

    def test_missing_processed_data_graceful(self):
        real_load = load_dataset
        def missing_processed(path, processed=False):
            if processed:
                raise DataError('Missing processed CSV')
            return real_load(path)
        with patch('src.dashboard_data.load_dataset', side_effect=missing_processed):
            app = self.app()
            app.sidebar.radio[0].set_value('Market Analysis').run()
            self.assertFalse(app.exception)
            self.assertTrue(app.error)


if __name__ == '__main__':
    unittest.main()
