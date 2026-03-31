"""
analysis.py
-----------
Stages 10-11: Regime analysis, robustness testing, and plotting.

All visualization and higher-level analysis lives here.
Keep matplotlib calls out of other modules.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

from xsec_alpha.metrics import (
    sharpe_ratio, max_drawdown, cumulative_returns,
    rolling_sharpe, ic_decay, metrics_by_regime
)


# ─────────────────────────────────────────────────────────────
# Plotting Helpers
# ─────────────────────────────────────────────────────────────

FIGURES_DIR = Path("reports/figures")


def save_fig(name: str):
    """Save current figure to reports/figures/."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIGURES_DIR / f"{name}.png", dpi=150, bbox_inches="tight")
    print(f"Saved: reports/figures/{name}.png")


# ─────────────────────────────────────────────────────────────
# Core Plots
# ─────────────────────────────────────────────────────────────

def plot_cumulative_returns(
    gross_returns: pd.Series,
    net_returns: pd.Series,
    save: bool = True,
):
    """Plot gross vs net cumulative returns over time."""
    fig, ax = plt.subplots(figsize=(12, 5))

    cumulative_returns(gross_returns).plot(ax=ax, label="Gross", color="steelblue")
    cumulative_returns(net_returns).plot(ax=ax, label="Net (10bps)", color="darkorange")

    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_title("Cumulative Portfolio Returns")
    ax.set_ylabel("Growth of $1")
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.tight_layout()

    if save:
        save_fig("cumulative_returns")
    plt.show()


def plot_rolling_sharpe(net_returns: pd.Series, window: int = 63, save: bool = True):
    """Plot rolling Sharpe ratio. Reveals regime-dependent performance."""
    fig, ax = plt.subplots(figsize=(12, 4))

    rs = rolling_sharpe(net_returns, window)
    rs.plot(ax=ax, color="steelblue", linewidth=1)

    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.axhline(1, color="green", linestyle=":", linewidth=0.8, label="Sharpe=1")
    ax.axhline(-1, color="red", linestyle=":", linewidth=0.8, label="Sharpe=-1")

    ax.set_title(f"Rolling {window}-Day Sharpe Ratio")
    ax.set_ylabel("Sharpe Ratio (annualized)")
    ax.legend()
    plt.tight_layout()

    if save:
        save_fig("rolling_sharpe")
    plt.show()


def plot_ic_over_time(ic_series: pd.Series, save: bool = True):
    """
    Plot IC time series with rolling mean.
    Key diagnostic: is predictive power stable or drifting?
    """
    fig, ax = plt.subplots(figsize=(12, 4))

    ic_series.plot(ax=ax, alpha=0.4, color="steelblue", label="Daily IC")
    ic_series.rolling(63).mean().plot(ax=ax, color="darkblue", linewidth=2, label="63-day rolling mean")

    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_title("Information Coefficient (IC) Over Time")
    ax.set_ylabel("Spearman IC")
    ax.legend()
    plt.tight_layout()

    if save:
        save_fig("ic_over_time")
    plt.show()


def plot_ic_decay(
    scores: pd.Series,
    prices: pd.DataFrame,
    horizons: list = [1, 5, 10, 20],
    save: bool = True,
):
    """
    Plot IC decay curve across forward return horizons.
    Reveals how long the signal's predictive power persists.
    """
    decay = ic_decay(scores, prices, horizons)

    fig, ax = plt.subplots(figsize=(8, 4))
    decay.plot(ax=ax, marker="o", color="steelblue")

    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_title("IC Decay: Signal Persistence Across Horizons")
    ax.set_xlabel("Forward Return Horizon (days)")
    ax.set_ylabel("Mean IC")
    ax.set_xticks(horizons)
    plt.tight_layout()

    if save:
        save_fig("ic_decay")
    plt.show()

    return decay


def plot_turnover(turnover: pd.Series, save: bool = True):
    """Plot daily turnover. High turnover = high costs = lower net returns."""
    fig, ax = plt.subplots(figsize=(12, 3))

    turnover.plot(ax=ax, alpha=0.6, color="steelblue")
    turnover.rolling(63).mean().plot(ax=ax, color="darkblue", linewidth=2, label="63-day avg")

    ax.set_title("Daily Portfolio Turnover")
    ax.set_ylabel("Turnover (fraction of portfolio)")
    ax.legend()
    plt.tight_layout()

    if save:
        save_fig("turnover")
    plt.show()


def plot_regime_performance(
    net_returns: pd.Series,
    regimes: pd.Series,
    save: bool = True,
):
    """
    Plot cumulative returns separately for high/low vol regimes.
    Key question: does the signal work in both market conditions?
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, regime in zip(axes, ["low_vol", "high_vol"]):
        mask = regimes == regime
        r = net_returns[mask].dropna()

        if len(r) == 0:
            continue

        cumulative_returns(r).plot(ax=ax, color="steelblue")
        ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
        ax.set_title(f"Regime: {regime.replace('_', ' ').title()}\n"
                     f"Sharpe: {sharpe_ratio(r):.2f} | MaxDD: {max_drawdown(r):.1%}")
        ax.set_ylabel("Growth of $1")

    plt.suptitle("Performance by Volatility Regime")
    plt.tight_layout()

    if save:
        save_fig("regime_performance")
    plt.show()


# ─────────────────────────────────────────────────────────────
# Robustness Testing
# ─────────────────────────────────────────────────────────────

def cost_sensitivity_table(
    gross_returns: pd.Series,
    turnover: pd.Series,
    cost_bps_levels: list = [5, 10, 20, 50],
) -> pd.DataFrame:
    """
    Compute Sharpe ratio at multiple cost levels.

    A robust signal maintains positive Sharpe even at high costs.
    If Sharpe collapses at 10bps, the signal is not tradeable.
    """
    from xsec_alpha.portfolio import apply_costs

    rows = []
    for bps in cost_bps_levels:
        net = apply_costs(gross_returns, turnover, bps)
        rows.append({
            "cost_bps": bps,
            "sharpe": round(sharpe_ratio(net), 3),
            "max_drawdown": round(max_drawdown(net), 4),
            "mean_daily_net": round(net.mean(), 6),
        })

    return pd.DataFrame(rows).set_index("cost_bps")


def plot_cost_sensitivity(
    gross_returns: pd.Series,
    turnover: pd.Series,
    cost_bps_levels: list = [5, 10, 20, 50],
    save: bool = True,
):
    """Plot how Sharpe ratio degrades as transaction costs increase."""
    table = cost_sensitivity_table(gross_returns, turnover, cost_bps_levels)

    fig, ax = plt.subplots(figsize=(8, 4))
    table["sharpe"].plot(ax=ax, marker="o", color="steelblue")

    ax.axhline(0, color="red", linestyle="--", linewidth=0.8, label="Sharpe=0")
    ax.set_title("Sharpe Ratio vs Transaction Cost Assumption")
    ax.set_xlabel("Transaction Cost (basis points)")
    ax.set_ylabel("Annualized Sharpe Ratio")
    ax.legend()
    plt.tight_layout()

    if save:
        save_fig("cost_sensitivity")
    plt.show()

    return table


def coefficient_stability_plot(
    coef_history: list,
    feature_names: list,
    save: bool = True,
):
    """
    Plot model coefficients across walk-forward windows.

    If coefficients flip signs wildly, the signal is unstable.
    Stable coefficients = interpretable, trustworthy model.

    Parameters
    ----------
    coef_history : list of pd.Series
        One Series per walk-forward window, from get_coefficients()
    feature_names : list
        Feature names for labeling
    """
    coef_df = pd.DataFrame(coef_history)

    fig, ax = plt.subplots(figsize=(12, 4))
    for col in coef_df.columns:
        coef_df[col].plot(ax=ax, marker="o", label=col)

    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_title("Model Coefficient Stability Across Walk-Forward Windows")
    ax.set_xlabel("Window Index")
    ax.set_ylabel("Coefficient Value")
    ax.legend()
    plt.tight_layout()

    if save:
        save_fig("coefficient_stability")
    plt.show()