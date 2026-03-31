"""
main.py
-------
Full pipeline orchestration. Reads like English.

Run with:
    python main.py

Each stage is clearly labeled. If something breaks, you know
exactly which stage to debug.
"""

import yaml
import pandas as pd
import numpy as np
from pathlib import Path

# ── Internal modules ──────────────────────────────────────────
from xsec_alpha.data.loader import load_prices, compute_returns, compute_forward_returns
from xsec_alpha.features import build_features
from xsec_alpha.validation import walk_forward_splits, print_splits
from xsec_alpha.model import prepare_xy, fit_predict, get_coefficients
from xsec_alpha.portfolio import (
    scores_to_weights, compute_gross_returns,
    compute_turnover, apply_costs, portfolio_summary
)
from xsec_alpha.metrics import (
    sharpe_ratio, max_drawdown, tail_stats,
    information_coefficient, ic_summary,
    newey_west_tstat, bootstrap_ic_ci,
    define_regimes, metrics_by_regime
)
from xsec_alpha.analysis import (
    plot_cumulative_returns, plot_rolling_sharpe,
    plot_ic_over_time, plot_ic_decay, plot_turnover,
    plot_regime_performance, plot_cost_sensitivity,
    coefficient_stability_plot, cost_sensitivity_table
)


def main():
    # ──────────────────────────────────────────────────────────
    # 0. Load Configuration
    # ──────────────────────────────────────────────────────────
    with open("configs/base.yaml") as f:
        cfg = yaml.safe_load(f)

    print("=" * 60)
    print("CROSS-SECTIONAL EQUITY SIGNAL RESEARCH")
    print("=" * 60)

    # ──────────────────────────────────────────────────────────
    # 1. Data Acquisition
    # ──────────────────────────────────────────────────────────
    print("\n[Stage 1] Loading price data...")
    prices = load_prices(cfg["tickers"], cfg["start_date"], cfg["end_date"])

    # ──────────────────────────────────────────────────────────
    # 2. Return Construction
    # ──────────────────────────────────────────────────────────
    print("\n[Stage 2] Computing returns...")
    returns = compute_returns(prices)
    fwd_returns = compute_forward_returns(prices, cfg["fwd_return_days"])
    print(f"Forward return horizon: {cfg['fwd_return_days']} days")

    # ──────────────────────────────────────────────────────────
    # 3. Feature Engineering
    # ──────────────────────────────────────────────────────────
    print("\n[Stage 3] Building features...")
    features = build_features(
        prices, returns,
        momentum_windows=cfg["momentum_windows"],
        volatility_window=cfg["volatility_window"],
    )
    from scipy import stats
    mom_flat = features["mom_20"].stack(future_stack=True).dropna()
    fwd_flat = fwd_returns.stack(future_stack=True).reindex(mom_flat.index).dropna()
    aligned = pd.concat([mom_flat, fwd_flat], axis=1).dropna()
    corr, pval = stats.spearmanr(aligned.iloc[:, 0], aligned.iloc[:, 1])
    print(f"\nRaw mom_20 vs fwd_return correlation: {corr:.4f} (p={pval:.4f})")

    # ──────────────────────────────────────────────────────────
    # 4-6. Walk-Forward Validation + Modeling
    # ──────────────────────────────────────────────────────────
    print("\n[Stage 4-6] Walk-forward validation + modeling...")
    print("\nValidation windows:")
    print_splits(
        prices.index,
        train_years=cfg["train_years"],
        test_months=cfg["test_months"],
        step_months=cfg["step_months"],
    )

    all_scores = []
    coef_history = []
    feature_names = list(features.keys())

    for i, (train_start, train_end, test_start, test_end) in enumerate(
        walk_forward_splits(
            prices.index,
            cfg["train_years"],
            cfg["test_months"],
            cfg["step_months"],
        )
    ):
        X_train, y_train = prepare_xy(features, fwd_returns, train_start, train_end)
        X_test, y_test   = prepare_xy(features, fwd_returns, test_start, test_end)

        # Skip windows with insufficient data
        if len(X_train) < 50 or len(X_test) < 5:
            print(f"  Window {i+1}: skipped (insufficient data)")
            continue

        scores, model = fit_predict(
            X_train, y_train, X_test,
            model_type="ridge",
            alpha=cfg["ridge_alpha"],
        )

        all_scores.append(scores)
        coef_history.append(get_coefficients(model, feature_names))

    all_scores = pd.concat(all_scores).sort_index()
    # Keep only the first prediction for each (date, ticker) pair.
    # Duplicates arise from overlapping walk-forward windows.
    all_scores = all_scores[~all_scores.index.duplicated(keep="first")]
    print(f"\nTotal scored observations: {len(all_scores):,}")
    coef_df = pd.DataFrame(coef_history, columns=feature_names)
    print("\nMean coefficients across windows:")
    print(coef_df.mean())
    print("\nCoefficient sign consistency (1.0 = always positive, 0.0 = always negative):")
    print((coef_df > 0).mean())

    # ──────────────────────────────────────────────────────────
    # 7-8. Portfolio Construction + Transaction Costs
    # ──────────────────────────────────────────────────────────
    print("\n[Stage 7-8] Building portfolio...")
    weights = scores_to_weights(
        all_scores,
        top_quantile=cfg["top_quantile"],
        bottom_quantile=cfg["bottom_quantile"],
    )

    gross_ret = compute_gross_returns(weights, returns)
    turnover  = compute_turnover(weights)
    net_ret   = apply_costs(gross_ret, turnover, cfg["default_cost_bps"])

    # Align to the period we have scores for
    gross_ret = gross_ret[gross_ret.index >= all_scores.index.get_level_values("date").min()]
    net_ret   = net_ret[net_ret.index >= all_scores.index.get_level_values("date").min()]
    turnover  = turnover[turnover.index >= all_scores.index.get_level_values("date").min()]

    print("\nPortfolio Summary:")
    summary = portfolio_summary(gross_ret, net_ret, turnover)
    for k, v in summary.items():
        print(f"  {k}: {v}")

    # ──────────────────────────────────────────────────────────
    # 9. Performance Metrics
    # ──────────────────────────────────────────────────────────
    print("\n[Stage 9] Computing performance metrics...")

    print(f"\nReturn Metrics:")
    print(f"  Sharpe (gross):        {sharpe_ratio(gross_ret):.3f}")
    print(f"  Sharpe (net {cfg['default_cost_bps']}bps):     {sharpe_ratio(net_ret):.3f}")
    print(f"  Max Drawdown:          {max_drawdown(net_ret):.2%}")

    tails = tail_stats(net_ret)
    print(f"\nTail Statistics:")
    for k, v in tails.items():
        print(f"  {k}: {v}")

    print(f"\nSignal Quality (IC Analysis):")
    ic = information_coefficient(all_scores, fwd_returns)
    ic_stats = ic_summary(ic)
    for k, v in ic_stats.items():
        print(f"  {k}: {v}")

    nw_t = newey_west_tstat(ic)
    ci_low, ci_high = bootstrap_ic_ci(ic)
    print(f"\nStatistical Significance:")
    print(f"  Newey-West t-stat:     {nw_t:.3f}")
    print(f"  Bootstrap 95% CI:      [{ci_low}, {ci_high}]")

    # ──────────────────────────────────────────────────────────
    # 10. Regime Analysis
    # ──────────────────────────────────────────────────────────
    print("\n[Stage 10] Regime analysis...")
    regimes = define_regimes(returns)
    regime_table = metrics_by_regime(net_ret, regimes)
    print("\nPerformance by Regime:")
    print(regime_table.to_string())

    # ──────────────────────────────────────────────────────────
    # 11. Robustness Testing
    # ──────────────────────────────────────────────────────────
    print("\n[Stage 11] Cost sensitivity analysis...")
    cost_table = cost_sensitivity_table(
        gross_ret, turnover,
        cost_bps_levels=cfg["cost_bps_levels"]
    )
    print("\nSharpe by Cost Level:")
    print(cost_table.to_string())

    # ──────────────────────────────────────────────────────────
    # Plots
    # ──────────────────────────────────────────────────────────
    print("\n[Plots] Generating charts...")
    plot_cumulative_returns(gross_ret, net_ret)
    plot_rolling_sharpe(net_ret)
    plot_ic_over_time(ic)
    plot_ic_decay(all_scores, prices)
    plot_turnover(turnover)
    plot_regime_performance(net_ret, regimes)
    plot_cost_sensitivity(gross_ret, turnover, cfg["cost_bps_levels"])
    coefficient_stability_plot(coef_history, feature_names)

    print("\n" + "=" * 60)
    print("Pipeline complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()