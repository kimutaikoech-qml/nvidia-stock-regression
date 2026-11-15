"""The single supported estimator and its coefficient interpretation."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_model():
    # Pipeline.fit learns both steps from the training partition only.
    return Pipeline([
        ("scaler", StandardScaler()),
        ("regression", LinearRegression()),
    ])


def coefficient_table(model):
    """Report USD per training standard deviation and USD per original unit."""
    scaler = model.named_steps['scaler']
    regression = model.named_steps['regression']
    return pd.DataFrame({
        'Feature': model.feature_names_in_,
        'Standardized coefficient': regression.coef_,
        'Original-unit coefficient': regression.coef_ / scaler.scale_,
    })


def original_intercept(model):
    scaler = model.named_steps['scaler']
    regression = model.named_steps['regression']
    return float(regression.intercept_ - np.dot(regression.coef_ / scaler.scale_, scaler.mean_))
