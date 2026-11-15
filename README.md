# NVIDIA Stock Price Prediction

A machine learning project that uses **Multiple Linear Regression** to predict NVIDIA (NVDA) next-trading-session adjusted closing prices. Built with Python, Scikit-learn, and Streamlit.

## Features

- Interactive four-page Streamlit dashboard
- Historical NVIDIA stock price and volume analysis
- Technical indicators: MA5, MA20, daily returns, and volatility
- On-demand next-session predictions using the saved model
- Historical prediction explorer with target-date selection and chart highlighting
- What-if simulator with editable inputs and training-range warnings
- Model performance evaluation and visualizations
- Regression coefficients and residual analysis

## Tech Stack

**Python • Pandas • NumPy • Scikit-learn • Streamlit • Plotly • yfinance • Joblib**

## Machine Learning Model

The model uses six features: Close, Volume, Daily Return, MA5, MA20, and 10-day Volatility.

**Pipeline:** `StandardScaler → LinearRegression`

**Training:** Chronological 80/20 train-test split.

### Model Performance

| Metric | Linear Regression | Baseline |
| ------ | ----------------: | -------: |
| MAE    |           $3.5327 |  $3.5365 |
| RMSE   |           $4.6140 |  $4.6033 |
| R²     |            0.9367 |        — |

The model performs similarly to the previous-close baseline, without demonstrating a meaningful predictive advantage. The test period was previously inspected during diagnostics, so these results are not from a fresh untouched holdout.

## Getting Started

**1. Clone the repository**
```bash
git clone https://github.com/kimutaikoech-qml/nvidia-stock-regression.git
cd nvidia-stock-regression
```

**2. Set up the environment**

Requires Python 3.14.4 (verified version).
```bash
python3.14 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**3. Run the dashboard**
```bash
streamlit run app.py
```

Open `http://localhost:8501`.

To regenerate the dataset and model:
```bash
python src/data.py
python src/features.py
python src/train.py
```

## Testing
```bash
python -m unittest discover -s tests -v
```
