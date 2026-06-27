from __future__ import annotations

import numpy as np
import pandas as pd


def _min_periods(window: int, min_periods: int | None) -> int:
    return window if min_periods is None else min_periods


def rolling_beta(
    target: pd.Series,
    benchmark: pd.Series,
    window: int,
    min_periods: int | None = None,
) -> pd.Series:
    """Rolling beta: covariance(target, benchmark) divided by benchmark variance."""
    target_aligned, benchmark_aligned = target.align(benchmark, join="outer")
    target_numeric = pd.to_numeric(target_aligned, errors="coerce")
    benchmark_numeric = pd.to_numeric(benchmark_aligned, errors="coerce")
    required = _min_periods(window, min_periods)
    covariance = target_numeric.rolling(window, min_periods=required).cov(benchmark_numeric)
    variance = benchmark_numeric.rolling(window, min_periods=required).var().replace(0, np.nan)
    return covariance / variance
