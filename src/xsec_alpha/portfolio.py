"""
portfolio.py
------------
Stages 7-8: Portfolio construction and transaction costs.

Takes predicted scores, builds weights, computes returns.
Applies a simple linear cost model.
"""

import pandas as pd
import numpy as np


def scores_to_weights(
    scores: pd.Series,
    top_quantile: float = 0.20,
    bottom_quantile: float = 0.0,
) -> pd.DataFrame:
    """
    Convert predicted scores to portfolio weights.

    For each date:
    - Long the top `top_quantile` fraction of tickers (equal weight)
    - Short the bottom `bottom_quantile` fraction (equal weight, negative)
    - Zero weight for everything in between

    Parameters
    ----------
    scores : pd.Series
        MultiIndex (date, ticker) — output of fit_predict
    top_quantile : float
        Fraction of tickers to go long (e.g. 0.2 = top 20%)
    bottom_quantile : float
        Fraction of tickers to go short (0.0 = long only)

    Returns
    -------
    pd.DataFrame
        Index: dates, Columns: tickers, Values: portfolio weights
        Long weights are positive, short weights are negative.
        Long weights sum to 1.0, short weights sum to -1.0 (if any).
    """
    # Unstack to wide format: dates x tickers
    scores_wide = scores.unstack(level="ticker")
    weights = pd.DataFrame(0.0, index=scores_wide.index, columns=scores_wide.columns)

    for date in scores_wide.index:
        row = scores_wide.loc[date].dropna()
        n = len(row)

        if n == 0:
            continue

        n_long = max(1, int(np.ceil(n * top_quantile)))
        n_short = int(np.floor(n * bottom_quantile))

        # Long: top n_long tickers
        long_tickers = row.nlargest(n_long).index
        weights.loc[date, long_tickers] = 1.0 / n_long

        # Short: bottom n_short tickers (if long-short mode)
        if n_short > 0:
            short_tickers = row.nsmallest(n_short).index
            weights.loc[date, short_tickers] = -1.0 / n_short

    return weights


def compute_gross_returns(
    weights: pd.DataFrame,
    returns: pd.DataFrame,
) -> pd.Series:
    """
    Compute daily portfolio return.

    portfolio_return_t = sum(weights_t * actual_returns_t)

    Parameters
    ----------
    weights : pd.DataFrame
        Dates x tickers (from scores_to_weights)
    returns : pd.DataFrame
        Daily simple returns, dates x tickers

    Returns
    -------
    pd.Series
        Daily gross portfolio returns
    """
    # Align columns
    common_tickers = weights.columns.intersection(returns.columns)
    w = weights[common_tickers]
    r = returns[common_tickers].reindex(w.index)

    # Element-wise multiply and sum across tickers
    portfolio_returns = (w * r).sum(axis=1)

    # Days with no weights (all zero) return 0 — that's correct
    return portfolio_returns


def compute_turnover(weights: pd.DataFrame) -> pd.Series:
    """
    Compute daily portfolio turnover.

    turnover_t = sum(|weights_t - weights_{t-1}|)

    Turnover of 1.0 means 100% of portfolio was traded that day.
    This directly drives transaction costs.

    Returns
    -------
    pd.Series
        Daily turnover
    """
    # Fill NaN with 0 (no position = 0 weight)
    w = weights.fillna(0.0)

    # Absolute change in weights from prior day
    daily_turnover = w.diff().abs().sum(axis=1)

    # First day has no prior — set to 0
    daily_turnover.iloc[0] = 0.0

    return daily_turnover


def apply_costs(
    gross_returns: pd.Series,
    turnover: pd.Series,
    cost_bps: float,
) -> pd.Series:
    """
    Apply linear transaction cost model.

    net_return_t = gross_return_t - cost_bps/10000 * turnover_t

    Parameters
    ----------
    gross_returns : pd.Series
        Daily gross portfolio returns
    turnover : pd.Series
        Daily turnover (from compute_turnover)
    cost_bps : float
        Cost per unit of turnover in basis points (1 bp = 0.01%)

    Returns
    -------
    pd.Series
        Daily net portfolio returns
    """
    cost_decimal = cost_bps / 10_000
    costs = turnover * cost_decimal
    return gross_returns - costs


def portfolio_summary(
    gross_returns: pd.Series,
    net_returns: pd.Series,
    turnover: pd.Series,
) -> dict:
    """Quick summary of portfolio-level stats."""
    return {
        "mean_daily_gross": round(gross_returns.mean(), 6),
        "mean_daily_net": round(net_returns.mean(), 6),
        "avg_daily_turnover": round(turnover.mean(), 4),
        "annualized_turnover": round(turnover.mean() * 252, 2),
        "n_trading_days": len(gross_returns),
    }