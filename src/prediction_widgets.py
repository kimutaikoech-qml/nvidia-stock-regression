"""Streamlit interactions for the saved NVIDIA model; no training or file writes."""
import logging

import streamlit as st

from src.dashboard_data import DataError
from src.model_artifacts import forecast_next_session
from src.model_config import FEATURES
from src.prediction_tools import (historical_result, interaction_token, latest_complete_row,
                                  simulate, training_ranges)


def prepare_interactions(bundle):
    try:
        latest = latest_complete_row(bundle['features'])
        token = interaction_token(bundle)
        ranges = training_ranges(bundle['features'], bundle['metrics'])
    except DataError as exc:
        logging.getLogger(__name__).warning('Interactive prediction unavailable: %s', exc)
        st.warning('Interactive inputs are unavailable for the current dataset.')
        return None
    if st.session_state.get('prediction_context') != token:
        # Clear stale estimates and reset control defaults when data/model changes.
        for key in list(st.session_state):
            if key.startswith('sim_input_') or key in ('generated_forecast', 'simulation_result', 'historical_target'):
                del st.session_state[key]
        st.session_state['prediction_context'] = token
    return latest, ranges


def forecast_widget(bundle, latest):
    st.subheader("Predict NVIDIA's Next Trading Session")
    row = latest.iloc[0]
    st.caption(f"Latest complete trading observation · {row.Date:%d %b %Y} · historical CSV snapshot, not live market data")
    if row.Date != bundle['features'].Date.max():
        st.warning('Newer rows have incomplete features; this estimate refers to the session after the last complete observation.')
    st.metric('Observation adjusted close', f'${row.Close:,.2f}')
    if st.button('Generate Prediction', key='generate_prediction', type='primary'):
        # Cached trusted Joblib loading happens in load_artifacts; only inference runs here.
        st.session_state.pop('generated_forecast', None)
        try:
            st.session_state['generated_forecast'] = forecast_next_session(bundle['model'], latest)
        except (DataError, ValueError, TypeError) as exc:
            logging.getLogger(__name__).warning('Forecast unavailable: %s', exc)
            st.warning('A forecast could not be generated from the available inputs.')
    forecast = st.session_state.get('generated_forecast')
    if forecast:
        cols = st.columns(3)
        cols[0].metric('Estimated next-session close', f"${forecast['predicted_close']:,.2f}")
        cols[1].metric('Estimated dollar change', f"${forecast['dollar_change']:+,.2f}")
        cols[2].metric('Estimated percentage change', f"{forecast['percent_change']:+.2f}%")
        if forecast['predicted_close'] <= 0:
            st.warning('The model returned a non-positive estimate; it is not economically meaningful.')
    st.caption(f"Estimate for the first trading session after {row.Date:%d %b %Y}; no calendar date is assumed. Model estimates are not guaranteed prices.")


def historical_widget(predictions):
    st.subheader('Historical Prediction Explorer')
    st.caption('Saved diagnostic test predictions · this test period was previously inspected. These are not live forecasts.')
    target = st.selectbox('Target trading date', predictions.Date.dt.date.tolist(),
                          index=len(predictions) - 1, format_func=lambda d: d.strftime('%d %b %Y'),
                          key='historical_target')
    row = historical_result(predictions, target)
    st.caption(f"Feature observation: {row.ObservationDate:%d %b %Y} → target trading session: {row.Date:%d %b %Y}")
    cols = st.columns(3)
    cols[0].metric('Historical predicted close', f'${row.Predicted:,.2f}')
    cols[1].metric('Historical actual close', f'${row.Actual:,.2f}')
    cols[2].metric('Prediction error · predicted − actual', f'${row.Predicted - row.Actual:+,.2f}')
    return row


def simulator_widget(bundle, latest, ranges):
    with st.expander('What-If Market Simulator'):
        st.write('Explore a hypothetical model response, not a real market forecast.')
        st.caption('Inputs are independent: changing Close does not recalculate moving averages, return, or volatility. '
                   'Some combinations may not describe a consistent price history. No model fitting occurs.')
        specs = [('Close', 'Closing price (USD)', 1., .01),
                 ('Volume', 'Trading volume (shares)', 1., 0.),
                 ('Daily_Return', 'Daily return (%)', 100., -99.99),
                 ('MA_5', 'Five-day moving average (USD)', 1., .01),
                 ('MA_20', 'Twenty-day moving average (USD)', 1., .01),
                 ('Volatility_10', 'Ten-day volatility (%)', 100., 0.)]
        inputs = {}
        with st.form('market_simulator'):
            columns = st.columns(2)
            for i, (feature, label, factor, minimum) in enumerate(specs):
                observed = bundle['features'][feature].dropna() * factor
                initial = float(latest[feature].iloc[0]) * factor
                maximum = max(float(observed.max()) * 3., initial * 3., 1.)
                if feature in ('Daily_Return', 'Volatility_10'):
                    maximum = max(maximum, 100.)
                step = 1000000. if feature == 'Volume' else .01
                with columns[i % 2]:
                    value = st.number_input(label, min_value=minimum, max_value=maximum, value=initial,
                                            step=step, format='%.0f' if feature == 'Volume' else '%.6f',
                                            key=f'sim_input_{feature}',
                                            help=f"Training range: {ranges.loc['min', feature] * factor:,.6g} to "
                                                 f"{ranges.loc['max', feature] * factor:,.6g}")
                inputs[feature] = value / factor
            submitted = st.form_submit_button('Run Simulation', type='primary')
        if submitted:
            st.session_state.pop('simulation_result', None)
            try:
                st.session_state['simulation_result'] = simulate(bundle['model'], inputs, ranges)
            except (DataError, ValueError, TypeError) as exc:
                st.warning(str(exc))
        result = st.session_state.get('simulation_result')
        if result:
            st.caption('Last submitted scenario · edit the controls and run again to update this result.')
            if result['outside_ranges']:
                st.warning('Outside historical training ranges: ' + ', '.join(result['outside_ranges']) +
                           '. This estimate extrapolates beyond the training examples.')
            submitted_inputs = result['inputs']
            train = bundle['features'].loc[bundle['features'].Date <= bundle['metrics']['train_end_date']]
            unusual = any(not (train[a] / train[b]).min() <= submitted_inputs[a] / submitted_inputs[b] <= (train[a] / train[b]).max()
                          for a, b in [('Close', 'MA_5'), ('MA_5', 'MA_20')])
            if unusual:
                st.warning('The price/moving-average combination is outside observed training patterns and may not describe a consistent market history.')
            cols = st.columns(3)
            cols[0].metric('Hypothetical next-session close', f"${result['predicted_close']:,.2f}")
            cols[1].metric('Hypothetical dollar change', f"${result['dollar_change']:+,.2f}")
            cols[2].metric('Hypothetical percentage change', f"{result['percent_change']:+.2f}%")
            if result['predicted_close'] <= 0:
                st.warning('The hypothetical estimate is non-positive and is not economically meaningful.')
            with st.expander('Submitted scenario'):
                st.json(submitted_inputs)
