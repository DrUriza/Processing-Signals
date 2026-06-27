from __future__ import annotations

import numpy as np
import pandas as pd

from processing_signals.processing.math.statistics import (
    classify_statistical_regime,
    rolling_autocorr,
    rolling_beta,
    rolling_drawdown,
    rolling_entropy,
    rolling_iqr,
    rolling_mean,
    rolling_std,
    rolling_zscore,
)


def test_rolling_moments_return_series_and_preserve_index() -> None:
    index = pd.date_range("2026-01-01", periods=5)
    series = pd.Series([1, 2, np.nan, 4, 5], index=index)

    mean = rolling_mean(series, window=2, min_periods=1)
    std = rolling_std(series, window=2, min_periods=1)
    zscore = rolling_zscore(series, window=2, min_periods=1)

    assert isinstance(mean, pd.Series)
    assert isinstance(std, pd.Series)
    assert isinstance(zscore, pd.Series)
    assert mean.index.equals(index)
    assert std.index.equals(index)
    assert zscore.index.equals(index)


def test_rolling_zscore_handles_zero_std() -> None:
    series = pd.Series([1, 1, 1, 1])
    out = rolling_zscore(series, window=2)
    assert out.isna().all()


def test_rolling_iqr_and_entropy() -> None:
    series = pd.Series([1, 2, 3, 4, np.nan, 6])

    iqr = rolling_iqr(series, window=3, min_periods=2)
    entropy = rolling_entropy(series, window=3, bins=3, min_periods=2)

    assert isinstance(iqr, pd.Series)
    assert isinstance(entropy, pd.Series)
    assert iqr.index.equals(series.index)
    assert entropy.index.equals(series.index)
    assert entropy.dropna().ge(0).all()


def test_rolling_autocorr_preserves_index() -> None:
    series = pd.Series([1, 2, 3, 4, 5])
    out = rolling_autocorr(series, window=3, lag=1, min_periods=2)
    assert isinstance(out, pd.Series)
    assert out.index.equals(series.index)


def test_rolling_drawdown_is_non_positive() -> None:
    series = pd.Series([10, 12, 9, 11, 8])
    out = rolling_drawdown(series, window=3, min_periods=1)
    assert isinstance(out, pd.Series)
    assert out.index.equals(series.index)
    assert out.dropna().le(0).all()


def test_rolling_beta_handles_zero_benchmark_variance() -> None:
    target = pd.Series([1, 2, 3, 4])
    benchmark = pd.Series([1, 1, 1, 1])
    out = rolling_beta(target, benchmark, window=2)
    assert isinstance(out, pd.Series)
    assert out.index.equals(target.index)
    assert out.isna().all()


def test_classify_statistical_regime_returns_dataframe() -> None:
    series = pd.Series([1, 2, 3, 4, 5, 100])
    out = classify_statistical_regime(series, window=3)
    assert isinstance(out, pd.DataFrame)
    assert out.index.equals(series.index)
    assert "stat_regime" in out.columns
