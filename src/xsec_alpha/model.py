"""
model.py
--------
Stage 5: Fit model, generate predicted scores.

We care about RANKING power, not absolute prediction accuracy.
A score of 0.8 vs 0.5 doesn't mean "this ETF returns 0.8%".
It means "this ETF is ranked higher than that one."

Start with Ridge regression. Later add Lasso, Random Forest for benchmarking.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from typing import Tuple

from xsec_alpha.features import stack_features


def prepare_xy(
    features_dict: dict,
    fwd_returns: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Slice features and target to a date range.
    Stack so each row is a (date, ticker) observation.
    Align features with target and drop any NaNs.

    Parameters
    ----------
    features_dict : dict
        Output of build_features()
    fwd_returns : pd.DataFrame
        Forward return target (dates x tickers)
    start, end : pd.Timestamp
        Date range to slice

    Returns
    -------
    X : pd.DataFrame
        Features, MultiIndex (date, ticker)
    y : pd.Series
        Target forward returns, same index as X
    """
    # Stack features to long format
    X = stack_features(features_dict, start, end)

    # Stack target to long format
    y = fwd_returns.loc[start:end].stack(future_stack=True)
    y.index.names = ["date", "ticker"]
    y.name = "target"

    # Align and drop rows with any NaN in features OR target
    combined = X.join(y, how="inner").dropna()

    X_clean = combined[list(features_dict.keys())]
    y_clean = combined["target"]

    return X_clean, y_clean


def fit_predict(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    model_type: str = "ridge",
    alpha: float = 1.0,
) -> Tuple[pd.Series, object]:
    """
    Fit a model on training data, predict scores on test data.

    Parameters
    ----------
    X_train, y_train : training features and target
    X_test : test features
    model_type : "ridge", "lasso", or "rf"
    alpha : regularization strength for Ridge/Lasso

    Returns
    -------
    scores : pd.Series
        Predicted scores with same index as X_test (MultiIndex date, ticker)
    model : fitted sklearn model
        Kept for coefficient inspection
    """
    # Select model
    if model_type == "ridge":
        model = Ridge(alpha=alpha)
    elif model_type == "lasso":
        model = Lasso(alpha=alpha, max_iter=5000)
    elif model_type == "rf":
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=3,       # shallow = less overfit
            random_state=42,
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    model.fit(X_train, y_train)

    scores = pd.Series(
        model.predict(X_test),
        index=X_test.index,
        name="score",
    )

    return scores, model


def get_coefficients(model, feature_names: list) -> pd.Series:
    """
    Extract coefficients from a linear model (Ridge or Lasso).
    Used to track coefficient stability across walk-forward windows.
    """
    if hasattr(model, "coef_"):
        return pd.Series(model.coef_, index=feature_names)
    else:
        return pd.Series(dtype=float)