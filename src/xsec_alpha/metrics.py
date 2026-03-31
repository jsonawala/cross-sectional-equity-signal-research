"""
metrics.py
----------
Stages 9-10: Performance metrics and regime analysis.

Every function here is a pure calculation — no data loading,
no model fitting. Input returns/scores, output numbers.

Think of each function as answering one skeptical question:
  sharpe_ratio     → "Is return worth the risk?"
  max_drawdown     → "What's the worst it got?"
  information_coefficient → "Does ranking actually predict returns?"
  ic_summary       → "Is the predictive power statistically real?"
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Tuple


# ─────────────────────────────────────────────────────────────
# Return-Based Metrics
# ─────────────────────────────────────────────────────────────

def sharpe_ratio(returns: pd.Series, periods_per_year: int = 252) -> float:
    """
    Annualized Sharpe ratio.

    Sharpe = (mean_daily_return * 252) / (std_daily_return * sqrt(252))
           = sqrt(252) * mean / std

    Assumes risk-free rate of 0 (reasonable for relative return strategies).
    """
    if returns.std() == 0 or returns.isna().all():
        return np.nan
    return float(np.sqrt(periods_per_year) * returns.mean() / returns.std())


def max_drawdown(returns: pd.Series) -> float:
    """
    Maximum peak-to-trough decline in cumulative returns.

    Always negative (or zero). E.g. -0.25 means worst drawdown was -25%.
    """
    cumulative = (1 + returns.fillna(0)).cumprod()
    rolling_max = cumulative.cummax()
    drawdown = (cumulative - rolling_max) / rolling_max
    return float(drawdown.min())


def expected_shortfall(returns: pd.Series, alpha: float = 0.05) -> float:
    """
    Expected Shortfall (CVaR) at confidence level alpha.

    Average return on the worst `alpha` fraction of days.
    More informative than VaR because it captures the average tail loss,
    not just the threshold.

    E.g. ES at 5% = average return on worst 5% of days.
    """
    threshold = returns.quantile(alpha)
    tail = returns[returns <= threshold]
    return float(tail.mean()) if len(tail) > 0 else np.nan


def tail_stats(returns: pd.Series) -> dict:
    """Skewness, kurtosis, and expected shortfall."""
    return {
        "skewness": round(float(returns.skew()), 4),
        "kurtosis": round(float(returns.kurtosis()), 4),  # excess kurtosis (normal=0)
        "expected_shortfall_5pct": round(expected_shortfall(returns, 0.05), 6),
    }


def cumulative_returns(returns: pd.Series) -> pd.Series:
    """Compound cumulative returns starting from 1.0."""
    return (1 + returns.fillna(0)).cumprod()


def rolling_sharpe(returns: pd.Series, window: int = 63) -> pd.Series:
    """
    Rolling Sharpe ratio over a `window`-day window.
    window=63 ≈ 1 quarter. Useful for visualizing regime changes.
    """
    roll_mean = returns.rolling(window).mean()
    roll_std = returns.rolling(window).std()
    return np.sqrt(252) * roll_mean / roll_std


# ─────────────────────────────────────────────────────────────
# Signal Quality: Information Coefficient
# ─────────────────────────────────────────────────────────────

def information_coefficient(
    scores: pd.Series,
    fwd_returns: pd.DataFrame,
    min_obs: int = 3,
) -> pd.Series:
    """
    Compute daily IC: Spearman rank correlation between predicted
    scores and actual forward returns, for each date.

    IC_t = SpearmanCorr(scores_t, fwd_returns_t)

    Why Spearman (rank correlation)?
    - We care about ranking, not absolute values
    - Robust to outlier returns
    - Standard in cross-sectional equity research

    Parameters
    ----------
    scores : pd.Series
        MultiIndex (date, ticker) predicted scores
    fwd_returns : pd.DataFrame
        Dates x tickers actual forward returns
    min_obs : int
        Minimum tickers required to compute IC (avoid degenerate correlations)

    Returns
    -------
    pd.Series
        IC value for each date
    """
    ic_dict = {}

    for date, group in scores.groupby(level="date"):
        # Get scores for this date (indexed by ticker)
        s = group.droplevel("date")

        # Get actual returns for this date
        if date not in fwd_returns.index:
            continue
        r = fwd_returns.loc[date].reindex(s.index)

        # Align and drop NaNs
        valid = pd.DataFrame({"score": s, "return": r}).dropna()

        if len(valid) < min_obs:
            continue

        corr, _ = stats.spearmanr(valid["score"], valid["return"])
        ic_dict[date] = corr

    return pd.Series(ic_dict, name="ic")


def ic_summary(ic_series: pd.Series) -> dict:
    """
    Summarize IC distribution with key statistics.

    ICIR (IC Information Ratio) = mean(IC) / std(IC)
    Analogous to Sharpe ratio but for signal quality.
    ICIR > 0.5 is considered good; > 1.0 is excellent.

    t-stat tests whether mean IC is significantly different from 0.
    t > 2.0 → statistically significant at ~5% level.
    """
    n = len(ic_series.dropna())
    mean_ic = ic_series.mean()
    std_ic = ic_series.std()
    icir = mean_ic / std_ic if std_ic > 0 else np.nan
    t_stat = mean_ic / (std_ic / np.sqrt(n)) if (std_ic > 0 and n > 0) else np.nan

    return {
        "n_observations": n,
        "mean_ic": round(mean_ic, 4),
        "std_ic": round(std_ic, 4),
        "icir": round(icir, 4),
        "t_stat": round(t_stat, 4),
        "pct_positive": round((ic_series > 0).mean(), 4),
        "min_ic": round(ic_series.min(), 4),
        "max_ic": round(ic_series.max(), 4),
    }


def ic_decay(
    scores: pd.Series,
    prices: pd.DataFrame,
    horizons: list = [1, 5, 10, 20],
) -> pd.Series:
    """
    Compute mean IC at multiple forward return horizons.

    Reveals how long the signal's predictive power persists.
    - Short decay → signal is short-lived, requires frequent trading
    - Slow decay → signal is persistent, cheaper to trade

    Parameters
    ----------
    scores : pd.Series
        MultiIndex (date, ticker) predicted scores
    prices : pd.DataFrame
        Price data (used to compute forward returns at each horizon)
    horizons : list
        List of forward return horizons in days

    Returns
    -------
    pd.Series
        Mean IC indexed by horizon
    """
    decay = {}
    for h in horizons:
        fwd_ret_h = prices.pct_change(h).shift(-h)
        ic_h = information_coefficient(scores, fwd_ret_h)
        decay[h] = ic_h.mean()
    return pd.Series(decay, name="mean_ic_by_horizon")


# ─────────────────────────────────────────────────────────────
# Statistical Significance
# ─────────────────────────────────────────────────────────────

def newey_west_tstat(ic_series: pd.Series, lags: int = 5) -> float:
    """
    Newey-West adjusted t-statistic for mean IC.

    Standard t-stats assume independence between observations.
    But IC values on consecutive days are correlated (overlapping
    forward return windows). Newey-West corrects for this by
    inflating the standard error — giving a more conservative (honest) result.

    lags=5 is standard for 5-day forward returns.
    """
    from statsmodels.stats.stattools import durbin_watson
    import statsmodels.api as sm

    clean = ic_series.dropna()
    if len(clean) < 10:
        return np.nan

    # OLS regression of IC on a constant, with Newey-West standard errors
    X = np.ones(len(clean))
    model = sm.OLS(clean.values, X)
    result = model.fit(cov_type="HAC", cov_kwds={"maxlags": lags})

    return float(result.tvalues[0])


def bootstrap_ic_ci(
    ic_series: pd.Series,
    n_bootstrap: int = 5000,
    confidence: float = 0.95,
) -> Tuple[float, float]:
    """
    Bootstrap confidence interval for mean IC.

    Resamples IC values with replacement to build an empirical
    distribution of the mean. Makes no distributional assumptions.

    Returns (lower_bound, upper_bound) of the confidence interval.
    """
    clean = ic_series.dropna().values
    n = len(clean)

    if n < 10:
        return (np.nan, np.nan)

    rng = np.random.default_rng(seed=42)
    bootstrap_means = [
        rng.choice(clean, size=n, replace=True).mean()
        for _ in range(n_bootstrap)
    ]

    alpha = 1 - confidence
    lower = np.percentile(bootstrap_means, 100 * alpha / 2)
    upper = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))

    return (round(lower, 4), round(upper, 4))


# ─────────────────────────────────────────────────────────────
# Regime Analysis
# ─────────────────────────────────────────────────────────────

def define_regimes(
    returns: pd.DataFrame,
    window: int = 20,
) -> pd.Series:
    """
    Define high/low volatility regimes using rolling market volatility.

    Uses SPY (first column or 'SPY' if present) as the market proxy.
    Regime = "high_vol" if rolling vol > median, else "low_vol".

    Returns
    -------
    pd.Series
        Index: dates, Values: "high_vol" or "low_vol"
    """
    # Use SPY as market proxy if available, else first column
    if "SPY" in returns.columns:
        market_returns = returns["SPY"]
    else:
        market_returns = returns.iloc[:, 0]

    rolling_vol = market_returns.rolling(window).std()
    median_vol = rolling_vol.median()

    regimes = pd.Series("low_vol", index=returns.index)
    regimes[rolling_vol > median_vol] = "high_vol"
    regimes[rolling_vol.isna()] = np.nan

    return regimes


def metrics_by_regime(
    returns: pd.Series,
    regimes: pd.Series,
) -> pd.DataFrame:
    """
    Compute Sharpe, max drawdown, and mean return for each regime.

    Returns
    -------
    pd.DataFrame
        Rows: regime names, Columns: metrics
    """
    results = {}
    for regime in regimes.dropna().unique():
        mask = regimes == regime
        r = returns[mask].dropna()
        results[regime] = {
            "n_days": len(r),
            "mean_daily_return": round(r.mean(), 6),
            "sharpe": round(sharpe_ratio(r), 4),
            "max_drawdown": round(max_drawdown(r), 4),
            "hit_rate": round((r > 0).mean(), 4),
        }

    return pd.DataFrame(results).T