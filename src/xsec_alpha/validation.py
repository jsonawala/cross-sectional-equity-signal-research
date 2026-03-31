"""
validation.py
-------------
Stage 4: Walk-forward validation splits.

No random splits. Always train on past, test on future.
This is the correct approach for time-series data.

          |---- Train 2yr ----|--- Test 3mo ---|
                              |---- Train 2yr ----|--- Test 3mo ---|
                                                  |---- Train 2yr ----|--- Test 3mo ---|
"""

import pandas as pd
from dateutil.relativedelta import relativedelta
from typing import Iterator, Tuple


def walk_forward_splits(
    index: pd.DatetimeIndex,
    train_years: int = 2,
    test_months: int = 3,
    step_months: int = 3,
) -> Iterator[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    """
    Generate (train_start, train_end, test_start, test_end) tuples.

    Parameters
    ----------
    index : pd.DatetimeIndex
        The full date index of your data
    train_years : int
        Length of each training window in years
    test_months : int
        Length of each test window in months
    step_months : int
        How far to roll forward each iteration

    Yields
    ------
    tuple of (train_start, train_end, test_start, test_end)
        All are pd.Timestamps
    """
    data_start = index.min()
    data_end = index.max()

    train_delta = relativedelta(years=train_years)
    test_delta = relativedelta(months=test_months)
    step_delta = relativedelta(months=step_months)

    train_start = data_start
    window_num = 0

    while True:
        train_end = train_start + train_delta
        test_start = train_end
        test_end = test_start + test_delta

        # Stop if test window exceeds available data
        if test_end > data_end:
            break

        window_num += 1
        yield (train_start, train_end, test_start, test_end)

        train_start += step_delta

    print(f"Walk-forward: generated {window_num} windows")


def print_splits(index: pd.DatetimeIndex, **kwargs):
    """
    Helper to visualize the splits. Call this to sanity-check
    your windows before running the full pipeline.
    """
    print(f"{'Window':<8} {'Train Start':<14} {'Train End':<14} {'Test Start':<14} {'Test End':<14}")
    print("-" * 70)
    for i, (ts, te, vs, ve) in enumerate(walk_forward_splits(index, **kwargs), 1):
        print(
            f"{i:<8} "
            f"{str(ts.date()):<14} "
            f"{str(te.date()):<14} "
            f"{str(vs.date()):<14} "
            f"{str(ve.date()):<14}"
        )