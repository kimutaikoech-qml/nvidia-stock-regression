"""Model evaluation and inference views backed only by verified local artifacts."""
import logging

import pandas as pd
import plotly.express as px
import streamlit as st

from src.dashboard_data import DataError
from src.dashboard_ui import COLORS, chart, header
from src.model_artifacts import forecast_next_session, load_artifacts
from src.model_config import FEATURES
from src.regression import coefficient_table, original_intercept


def _load():
    try:
        return load_artifacts()
    except DataError as exc:
        # Keep validation details in server logs rather than the financial UI.
        logging.getLogger(__name__).warning("Model results unavailable: %s", exc)
        st.warning("Model results are unavailable. Please prepare the saved model and evaluation results.")
        st.info("Run the training script from the project root to prepare model results.")
        return None


def comparison(metrics):
    return pd.DataFrame({
        'Method': ['Multiple Linear Regression', 'Naive · previous close'],
        'MAE (USD)': [metrics['model_mae'], metrics['baseline_mae']],
        'RMSE (USD)': [metrics['model_rmse'], metrics['baseline_rmse']],
    })


def predictions():
    header("NVDA / Prediction workspace", "ML Predictions",
           "Multiple Linear Regression · next-session adjusted closing-price estimates.")
    bundle = _load()
    if bundle is None:
        return
    m, p = bundle['metrics'], bundle['predictions']
    st.caption("🟢 Model trained")
    _forecast(bundle)

    st.subheader("Model evaluation")
    scores, context = st.columns([3, 2])
    with scores:
        cols = st.columns(3)
        cols[0].metric("Model MAE · USD", f"${m['model_mae']:.4f}")
        cols[1].metric("Model RMSE · USD", f"${m['model_rmse']:.4f}")
        cols[2].metric("Model R²", f"{m['model_r2']:.4f}")
    with context:
        st.info("The test period was previously inspected during diagnostics. "
                "The model performs similarly to the previous-close baseline.")

    st.subheader("Actual vs predicted adjusted close")
    st.caption(f"Test period · {p['Date'].min():%d %b %Y} — {p['Date'].max():%d %b %Y}")
    fig = px.line(p, x='Date', y=['Actual', 'Predicted', 'Baseline'],
                  color_discrete_sequence=COLORS,
                  labels={'Date': 'Target trading date', 'value': 'Adjusted close (USD)', 'variable': 'Series'})
    fig.update_traces(line_dash='dot', selector={'name': 'Baseline'})
    chart(fig, 'model_test_prices', height=420)


def _forecast(bundle):
    st.subheader("Next-trading-session forecast")
    try:
        forecast = forecast_next_session(bundle['model'], bundle['features'])
    except DataError as exc:
        logging.getLogger(__name__).warning("Forecast unavailable: %s", exc)
        st.warning("A forecast is unavailable for the current data.")
        return
    st.caption(f"LATEST COMPLETE OBSERVATION · {forecast['observation_date']:%d %b %Y}")
    if forecast['observation_date'] != forecast['dataset_latest_date']:
        st.warning("The most recent dataset rows have incomplete features. The estimate uses the last complete "
                   "observation and refers to the session after that observation.")
    cols = st.columns(4)
    cols[0].metric("Observation adjusted close", f"${forecast['latest_close']:,.2f}")
    cols[1].metric("Estimated next-session close", f"${forecast['predicted_close']:,.2f}")
    cols[2].metric("Estimated dollar change", f"${forecast['dollar_change']:+,.2f}")
    cols[3].metric("Estimated percentage change", f"{forecast['percent_change']:+.2f}%")
    st.caption(f"Estimate for the first trading session after {forecast['observation_date']:%d %b %Y} "
               "· historical snapshot · not a guaranteed price.")
    if forecast['predicted_close'] <= 0:
        st.warning("The unconstrained regression returned a non-positive price estimate; it is not economically meaningful.")


def insights():
    header("NVDA / Model transparency", "Model Insights",
           "Inspect coefficients, test residuals, and the previous-close benchmark.")
    bundle = _load()
    if bundle is None:
        return
    model, m, p = bundle['model'], bundle['metrics'], bundle['predictions']
    regression = model.named_steps['regression']
    st.subheader("Model setup")
    st.caption("StandardScaler → LinearRegression")
    st.write("The scaler is fitted only on the chronological training partition. "
             "Test observations and forecasts use the saved training transformation.")
    if regression.rank_ < len(FEATURES):
        st.warning("The fitted model has reduced numerical rank. Review feature scales and correlated inputs.")
    cols = st.columns(3)
    cols[0].metric("Effective numerical rank", f"{regression.rank_}/{len(FEATURES)}")
    cols[1].metric("Standardized intercept · USD", f"{float(regression.intercept_):.6g}")
    cols[2].metric("Original-unit intercept · USD", f"{original_intercept(model):.6g}")
    st.subheader("Feature coefficients")
    coefficients = coefficient_table(model)
    left, right = st.columns([1, 2])
    with left:
        st.dataframe(coefficients, hide_index=True, width='stretch',
                     column_config={name: st.column_config.NumberColumn(format='%.8g')
                                    for name in ['Standardized coefficient', 'Original-unit coefficient']})
    with right:
        chart(px.bar(coefficients, x='Standardized coefficient', y='Feature', orientation='h',
                     color_discrete_sequence=COLORS,
                     labels={'Standardized coefficient': 'USD per training standard deviation'}), 'model_coefficients', height=320)
    st.info("Standardized coefficients show the USD change per one training standard deviation, holding other "
            "inputs fixed. Original-unit coefficient magnitudes are not directly comparable because units differ "
            "(USD, shares, and fractional returns). Correlated moving averages can still make coefficients unstable; "
            "these values are not a causal feature-importance ranking.")
    st.subheader("Prediction residuals")
    st.caption("Residual = actual adjusted close − predicted adjusted close, in USD. Positive values indicate underprediction.")
    residual_fig = px.scatter(p, x='Date', y='Residual', color_discrete_sequence=COLORS,
                              labels={'Date': 'Target trading date', 'Residual': 'Residual (USD)'})
    residual_fig.add_hline(y=0, line_dash='dash', line_color=COLORS[1])
    chart(residual_fig, 'model_residuals')
    st.subheader("Actual vs predicted scatterplot")
    scatter = px.scatter(p, x='Actual', y='Predicted', hover_data=['Date', 'ObservationDate'],
                         color_discrete_sequence=COLORS,
                         labels={'Actual': 'Actual adjusted close (USD)', 'Predicted': 'Predicted adjusted close (USD)'})
    low, high = p[['Actual', 'Predicted']].min().min(), p[['Actual', 'Predicted']].max().max()
    scatter.add_shape(type='line', x0=low, y0=low, x1=high, y1=high,
                      line=dict(color=COLORS[1], dash='dash'))
    chart(scatter, 'model_scatter')
    st.caption("Dashed line: perfect agreement. All points come from the held-out test set.")
    st.subheader("Model vs naive baseline")
    table = comparison(m)
    st.dataframe(table, hide_index=True, width='stretch')
    chart(px.bar(table.melt(id_vars='Method', var_name='Metric', value_name='Error (USD)'),
                 x='Metric', y='Error (USD)', color='Method', barmode='group',
                 color_discrete_sequence=COLORS), 'model_baseline_comparison', height=300)
    with st.expander("Evaluation details", expanded=True):
        st.info(m['evaluation_note'])
        cols = st.columns(4)
        cols[0].metric("Baseline MAE · USD", f"${m['baseline_mae']:.4f}")
        cols[1].metric("Baseline RMSE · USD", f"${m['baseline_rmse']:.4f}")
        cols[2].metric("Training samples", f"{m['train_samples']:,}")
        cols[3].metric("Testing samples", f"{m['test_samples']:,}")
        st.caption(f"TARGET DATES · Last training target: {m['train_target_end_date']} · "
                   f"Test targets: {m['test_target_start_date']} — {m['test_target_end_date']}")
        st.write(f"Chronological split: {m['train_samples']:,} training samples and {m['test_samples']:,} test samples. "
                 "The target is the next observed trading session’s adjusted close. "
                 "The naive baseline predicts that session’s close equals the input observation’s close.")
        st.caption(f"Training observation end: {m['train_end_date']} · Test observation start: {m['test_start_date']} · "
                   f"Test target end: {m['test_target_end_date']} · scikit-learn {m['sklearn_version']}")
        st.write("The saved model is fitted on the training partition only. Forecasting loads that model without "
                 "refitting on the test set. Backtest errors do not guarantee future forecast accuracy.")

    with st.expander("Forecast inputs and timing"):
        latest = bundle['features'].dropna(subset=FEATURES).sort_values('Date').tail(1)
        st.dataframe(latest[['Date'] + FEATURES], hide_index=True, width='stretch')
        st.write("Inputs follow the exact training feature order and use only the observation date and "
                 "preceding observations. No future closing price is an input.")
        st.write("The forecast is for the next trading session after the latest complete observation. "
                 "An exact future date is not assigned because no exchange holiday/session calendar is configured.")
