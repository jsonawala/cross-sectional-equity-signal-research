"""
features.py
-----------
Stage 3: Feature engineering.

Each function takes prices or returns and returns a DataFrame
of the same shape (dates x tickers).

Cross-sectional z-scoring is applied so that on each date,
each ticker's feature value is expressed relative to other tickers
that day — not relative to its own history.
"""

import pandas as pd
import numpy as np


# ─────────────────────────────────────────────────────────────
# Raw Feature Calculations
# ─────────────────────────────────────────────────────────────

def momentum(prices: pd.DataFrame, window: int) -> pd.DataFrame:
    """
    Return over the last `window` trading days.

    momentum_t = P_t / P_{t-window} - 1

    Higher value = stronger recent uptrend.
    """
    return prices.pct_change(window)


def volatility(returns: pd.DataFrame, window: int) -> pd.DataFrame:
    """
    Rolling standard deviation of daily returns over `window` days.

    Higher value = more volatile / uncertain asset.
    Note: high vol can be a risk signal OR a momentum signal depending
    on the regime. We let the model figure out the relationship.
    """
    return returns.rolling(window=window, min_periods=int(window * 0.8)).std()


# ─────────────────────────────────────────────────────────────
# Cross-Sectional Normalization
# ─────────────────────────────────────────────────────────────

def cross_sectional_zscore(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each date (row), z-score the values across tickers (columns).

    z_it = (x_it - mean_t) / std_t

    where mean_t and std_t are computed across all tickers on date t.

    Why this matters:
    - Removes time-varying level effects (e.g., all momentum high in bull markets)
    - Makes features comparable across different market regimes
    - Model sees RELATIVE attractiveness, not absolute values

    Rows with std = 0 (all tickers identical) become NaN — this is correct.
    """
    mean = df.mean(axis=1)   # mean across tickers each day
    std = df.std(axis=1)     # std across tickers each day

    # Subtract row mean, divide by row std
    zscored = df.subtract(mean, axis=0).divide(std, axis=0)

    return zscored


# ─────────────────────────────────────────────────────────────
# Master Feature Builder
# ─────────────────────────────────────────────────────────────

def build_features(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    momentum_windows: list = [20, 60],
    volatility_window: int = 20,
) -> dict:
    """
    Build all features and return as a dictionary of DataFrames.

    Each DataFrame has shape: (dates x tickers)
    Each value is a cross-sectionally z-scored feature.

    Parameters
    ----------
    prices : pd.DataFrame
        Adjusted close prices
    returns : pd.DataFrame
        Daily simple returns (from compute_returns)
    momentum_windows : list
        Lookback windows for momentum features
    volatility_window : int
        Lookback window for volatility feature

    Returns
    -------
    dict
        Keys: feature names (e.g. "mom_20", "mom_60", "vol_20")
        Values: DataFrames of shape (dates x tickers)
    """
    features = {}

    # Momentum features
    for w in momentum_windows:
        raw = momentum(prices, w)
        features[f"mom_{w}"] = cross_sectional_zscore(raw)
        print(f"Built feature: mom_{w}")

    # Volatility feature
    raw_vol = volatility(returns, volatility_window)
    features[f"vol_{volatility_window}"] = cross_sectional_zscore(raw_vol)
    print(f"Built feature: vol_{volatility_window}")

    # Log feature coverage (how many non-NaN values exist)
    for name, df in features.items():
        coverage = df.notna().mean().mean()
        print(f"  {name} coverage: {coverage:.1%}")

    return features


def stack_features(
    features_dict: dict,
    start: pd.Timestamp,
    end: pd.Timestamp
) -> pd.DataFrame:
    """
    Slice features to a date range and stack into a long-format DataFrame.

    Output shape: (dates * tickers, n_features)
    Index: MultiIndex (date, ticker)

    This is the format sklearn expects: one row per observation.
    """
    frames = []
    for name, df in features_dict.items():
        sliced = df.loc[start:end].stack(future_stack=True)
        sliced.name = name
        frames.append(sliced)

    stacked = pd.concat(frames, axis=1)
    stacked.index.names = ["date", "ticker"]

    return stacked