from __future__ import annotations

import pandas as pd


def _min_periods(window: int, min_periods: int | None) -> int:
    return window if min_periods is None else min_periods


def rolling_autocorr(
    series: pd.Series,
    window: int,
    lag: int = 1,
    min_periods: int | None = None,
) -> pd.Series:
    """Rolling autocorrelation against a lagged copy."""
    s = pd.to_numeric(series.copy(), errors="coerce")
    return s.rolling(window, min_periods=_min_periods(window, min_periods)).corr(s.shift(lag))


def rolling_correlation(
    series_a: pd.Series,
    series_b: pd.Series,
    window: int,
    min_periods: int | None = None,
) -> pd.Series:
    """Rolling correlation after index alignment."""
    a, b = series_a.align(series_b, join="outer")
    a = pd.to_numeric(a, errors="coerce")
    b = pd.to_numeric(b, errors="coerce")
    return a.rolling(window, min_periods=_min_periods(window, min_periods)).corr(b)
