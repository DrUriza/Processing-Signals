from __future__ import annotations

import numpy as np
import pandas as pd


def _min_periods(window: int, min_periods: int | None) -> int:
    return window if min_periods is None else min_periods


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.copy(), errors="coerce")


def rolling_quantile(
    series: pd.Series,
    window: int,
    quantile: float,
    min_periods: int | None = None,
) -> pd.Series:
    """Rolling quantile."""
    return _numeric(series).rolling(window, min_periods=_min_periods(window, min_periods)).quantile(quantile)


def rolling_min(series: pd.Series, window: int, min_periods: int | None = None) -> pd.Series:
    """Rolling minimum."""
    return _numeric(series).rolling(window, min_periods=_min_periods(window, min_periods)).min()


def rolling_max(series: pd.Series, window: int, min_periods: int | None = None) -> pd.Series:
    """Rolling maximum."""
    return _numeric(series).rolling(window, min_periods=_min_periods(window, min_periods)).max()


def rolling_range(series: pd.Series, window: int, min_periods: int | None = None) -> pd.Series:
    """Rolling range: max minus min."""
    return rolling_max(series, window, min_periods) - rolling_min(series, window, min_periods)


def rolling_iqr(series: pd.Series, window: int, min_periods: int | None = None) -> pd.Series:
    """Rolling interquartile range."""
    q75 = rolling_quantile(series, window, 0.75, min_periods)
    q25 = rolling_quantile(series, window, 0.25, min_periods)
    return q75 - q25


def rolling_entropy(
    series: pd.Series,
    window: int,
    bins: int = 10,
    min_periods: int | None = None,
) -> pd.Series:
    """Rolling Shannon entropy from a histogram in each window."""
    s = _numeric(series)
    required = _min_periods(window, min_periods)

    def entropy(values: np.ndarray) -> float:
        clean = values[~np.isnan(values)]
        if len(clean) < required:
            return np.nan
        counts, _ = np.histogram(clean, bins=bins)
        total = counts.sum()
        if total == 0:
            return np.nan
        probabilities = counts[counts > 0] / total
        return float(-(probabilities * np.log(probabilities)).sum())

    return s.rolling(window, min_periods=required).apply(entropy, raw=True)
