# NVIDIA Stock Price Prediction Using Linear Regression

A machine learning-powered stock analytics dashboard built with **Python, Scikit-learn, Pandas, Plotly, and Streamlit** to analyze NVIDIA (NASDAQ: NVDA) historical stock performance and estimate next-trading-session adjusted closing prices.

The project demonstrates an end-to-end machine learning workflow, from historical data collection and feature engineering to regression modeling, evaluation, visualization, and an interactive application.

## Project Overview

This project investigates whether Multiple Linear Regression can predict NVIDIA's next-session adjusted closing price using historical market data and technical indicators.

Beyond price prediction, the project emphasizes sound machine learning practices, including chronological data splitting, feature scaling, numerical stability, baseline comparisons, and transparent evaluation.

### Key Features

- **Interactive Dashboard:** Four-page Streamlit application with a modern NVIDIA-inspired dark theme.
- **Historical Market Analysis:** Visualize adjusted stock prices, trading volume, daily returns, and volatility.
- **Feature Engineering:** Calculate moving averages and statistical indicators from historical prices.
- **Multiple Linear Regression:** Predict next-session adjusted closing prices using six input features.
- **Model Evaluation:** Evaluate MAE, RMSE, and R² against a previous-close baseline.
- **Model Insights:** Explore regression coefficients, residuals, numerical rank, and prediction errors.
- **Reproducible Pipeline:** Save and reuse the trained preprocessing and regression pipeline using Joblib.

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Pandas | Data manipulation and preprocessing |
| NumPy | Numerical calculations |
| Scikit-learn | Feature scaling and Linear Regression |
| yfinance | Historical NVIDIA stock data |
| Streamlit | Interactive web dashboard |
| Plotly | Interactive financial visualizations |
| Matplotlib | Evaluation plots |
| Joblib | Model serialization |
| unittest | Automated testing |

## Project Structure

```text
nvidia-stock-regression/
├── app.py
├── .streamlit/
│   └── config.toml
├── .python-version
├── data/
│   ├── raw/
│   │   └── nvda.csv
│   └── processed/
│       └── nvda_features.csv
├── src/
│   ├── data.py
│   ├── features.py
│   ├── train.py
│   ├── regression.py
│   ├── model_config.py
│   ├── model_artifacts.py
│   ├── model_pages.py
│   ├── dashboard_data.py
│   ├── dashboard_pages.py
│   ├── dashboard_ui.py
│   └── dashboard.css
├── models/
│   └── linear_regression.pkl
├── results/
│   ├── metrics.json
│   ├── predictions.csv
│   └── figures/
│       └── predictions.png
├── tests/
│   ├── test_dashboard.py
│   └── test_model.py
├── requirements.txt
├── .gitignore
└── README.md
```

This structure matches the current project. Data collection is implemented in `src/data.py`; training and evaluation are both implemented in `src/train.py`. File paths are derived from `__file__` using `pathlib`.

## Machine Learning Methodology

### 1. Data Collection

Historical NVIDIA (NVDA) stock data is retrieved using the `yfinance` Python library, beginning January 1, 2020.

The dataset includes:

- Open, High, Low, and Close prices
- Trading volume
- Trading dates

Adjusted historical prices are used to account for relevant corporate actions.

### 2. Feature Engineering

Six input features are used to train the model:

| Feature | Description |
|---|---|
| Close | Current session's adjusted closing price |
| Volume | Daily trading volume |
| Daily_Return | Fractional change in closing price (`0.01` means 1%) |
| MA_5 | Five-session moving average |
| MA_20 | Twenty-session moving average |
| Volatility_10 | Ten-session rolling standard deviation of returns |

The prediction target is defined as the next trading session's adjusted closing price.

```python
df["Target"] = df["Close"].shift(-1)
```

All predictive features are calculated using information available at or before the current trading session.

### 3. Model Architecture

The project uses a Scikit-learn pipeline:

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

model = Pipeline([
    ("scaler", StandardScaler()),
    ("regression", LinearRegression())
])
```

StandardScaler standardizes the input features using statistics learned from the training data.

LinearRegression estimates the relationship between the input features and the next-session closing price using ordinary least squares.

The corrected model achieves an effective numerical rank of **6/6**, retaining all six feature dimensions.

### 4. Training and Evaluation

The historical dataset is divided chronologically:

- 80% for training
- 20% for testing

The model is evaluated using:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Coefficient of Determination (R²)

A naive previous-close baseline is used for comparison:

**Predicted next-session close = Current-session close**

This baseline helps determine whether the regression model provides predictive value beyond price persistence.

## Model Results

The corrected Linear Regression model produced the following results:

| Metric | Linear Regression | Previous-Close Baseline |
|---|---:|---:|
| MAE | $3.5327 | $3.5365 |
| RMSE | $4.6140 | $4.6033 |
| R² | 0.9367 | Not reported |
| Numerical Rank | 6/6 | N/A |

### Interpretation

The Linear Regression model achieves a high R² and relatively low prediction errors compared with the observed price scale.

However, its performance is nearly identical to the previous-close baseline:

- Linear Regression achieves a slightly lower MAE.
- The previous-close baseline achieves a slightly lower RMSE.
- The results do not establish a reliable predictive or trading advantage.

**Evaluation limitation:** The chronological test period was inspected during numerical diagnostics. These results are therefore reported as a diagnostic comparison rather than an untouched final holdout. No model parameters were optimized against this test period.

## Example Next-Session Prediction

Using the November 13, 2026 observation:

| Measure | Value |
|---|---:|
| Observed adjusted close | $162.85 |
| Predicted next-session close | $164.16 |
| Estimated change | +$1.31 |
| Estimated percentage change | +0.80% |

This is a snapshot-based model estimate, not a guaranteed future price or investment recommendation.

## Streamlit Dashboard

The application contains four interactive sections.

### Overview

Explore NVIDIA's historical stock prices, trading volume, summary metrics, and historical trading records.

### Market Analysis

Analyze short-term and longer-term moving averages, daily returns, volatility, and statistical distributions.

### ML Predictions

View next-session forecasts, actual versus predicted closing prices, and MAE/RMSE/R² metrics, with a concise note about the previously inspected test period and similar baseline performance.

### Model Insights

Examine preprocessing, model coefficients, residuals, effective numerical rank, detailed baseline comparisons, chronological split details, and forecast inputs.

## Installation and Setup

### Prerequisites

- Python **3.14.4**, the verified environment recorded in `.python-version` (other Python versions have not been verified)
- Git
- Internet access for downloading historical stock data

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/nvidia-stock-regression.git
cd nvidia-stock-regression
```

Replace `YOUR_USERNAME` with your GitHub username.

### 2. Create a Virtual Environment

```bash
python3.14 -m venv venv
source venv/bin/activate
```

For Windows, with a compatible Python installation (the project has been verified on Ubuntu):

```powershell
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
python -m pip install -r requirements.txt
python -m pip check
```

`requirements.txt` pins the runtime dependencies from the verified environment. A fresh download/install has not been independently verified.

The included CSVs and trained pipeline can be used directly. To view the existing dashboard, skip steps 4–6 and launch it using step 7. Only refresh data or retrain when you deliberately want to regenerate the experiment; refreshed data can change the results and forecast example above.

### 4. Download Historical Stock Data (Optional)

```bash
python src/data.py
```

### 5. Generate Features (After Refreshing Data)

```bash
python src/features.py
```

### 6. Train the Model (When Regenerating Artifacts)

```bash
python src/train.py
```

This trains the complete StandardScaler–LinearRegression pipeline and generates model and evaluation artifacts. Streamlit loads the saved pipeline and does not retrain on refresh. Only load Joblib files generated by your trusted local training script.

### 7. Launch the Dashboard

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

## Testing

Run the project's automated test suite:

```bash
python -m unittest discover -s tests -v
```

The suite uses Python's built-in `unittest` and Streamlit AppTest to verify the data pipeline, feature engineering, training-only scaling, feature ordering, model behavior, and dashboard integration. AppTest is not a real-browser visual test.

## Limitations

This project has several limitations:

1. **Market uncertainty:** Historical stock prices do not fully capture earnings announcements, macroeconomic events, or unexpected market developments.
2. **Linear assumptions:** The model cannot directly represent nonlinear relationships without additional feature transformations.
3. **Baseline performance:** The model has not demonstrated a meaningful advantage over a previous-close prediction.
4. **Evaluation reuse:** The existing test period was previously inspected during diagnostics.
5. **Execution realism:** The project predicts adjusted closing prices but does not simulate transaction costs, slippage, or executable trading strategies.
6. **Data availability:** Historical market data availability and adjustments depend on the data provider.

## Future Improvements

Potential future work includes:

- Walk-forward validation
- Evaluation on newly collected, untouched market data
- Ridge and Lasso Regression
- Additional economic and market features
- Prediction of next-session returns
- Uncertainty estimation and prediction intervals
- Comparison with nonlinear machine learning models
- Automated data refresh and model monitoring

Any new modeling approach should be evaluated using a clearly defined validation protocol to reduce the risk of overfitting.

## Disclaimer

This project is intended for educational and research purposes only.

Stock price predictions are uncertain and should not be interpreted as financial advice or guaranteed trading signals.

## Author

**Kipngeno Koech**

Machine Learning | Artificial Intelligence | Financial Data Analytics
