from __future__ import annotations

import numpy as np
import pandas as pd


def _min_periods(window: int, min_periods: int | None) -> int:
    return window if min_periods is None else min_periods


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.copy(), errors="coerce")


def rolling_mean(series: pd.Series, window: int, min_periods: int | None = None) -> pd.Series:
    """Rolling arithmetic mean."""
    return _numeric(series).rolling(window, min_periods=_min_periods(window, min_periods)).mean()


def rolling_std(series: pd.Series, window: int, min_periods: int | None = None) -> pd.Series:
    """Rolling sample standard deviation."""
    return _numeric(series).rolling(window, min_periods=_min_periods(window, min_periods)).std()


def rolling_var(series: pd.Series, window: int, min_periods: int | None = None) -> pd.Series:
    """Rolling sample variance."""
    return _numeric(series).rolling(window, min_periods=_min_periods(window, min_periods)).var()


def rolling_skewness(series: pd.Series, window: int, min_periods: int | None = None) -> pd.Series:
    """Rolling skewness."""
    return _numeric(series).rolling(window, min_periods=_min_periods(window, min_periods)).skew()


def rolling_kurtosis(series: pd.Series, window: int, min_periods: int | None = None) -> pd.Series:
    """Rolling kurtosis."""
    return _numeric(series).rolling(window, min_periods=_min_periods(window, min_periods)).kurt()


def rolling_zscore(series: pd.Series, window: int, min_periods: int | None = None) -> pd.Series:
    """Rolling z-score using rolling mean and standard deviation."""
    s = _numeric(series)
    mean = rolling_mean(s, window, min_periods)
    std = rolling_std(s, window, min_periods).replace(0, np.nan)
    return (s - mean) / std
