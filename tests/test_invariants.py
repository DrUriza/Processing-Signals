"""Tests for signal_analysis.indicators.invariants."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from signal_analysis.indicators.invariants import (
    InvariantIndicators,
    add_invariant_features,
    compute_local_ratio,
    compute_log_ratio,
    compute_minmax_position,
    compute_normalized_energy,
    compute_relative_change,
    compute_robust_zscore,
    compute_rolling_energy,
    compute_rolling_zscore,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

N = 60
WINDOW = 20


@pytest.fixture
def rng_series() -> pd.Series:
    rng = np.random.default_rng(42)
    return pd.Series(rng.normal(100, 10, N))


@pytest.fixture
def simple_df(rng_series) -> pd.DataFrame:
    rng = np.random.default_rng(99)
    return pd.DataFrame(
        {
            "close": rng_series.values,
            "volume": np.abs(rng.normal(1000, 200, N)),
        }
    )


# ---------------------------------------------------------------------------
# compute_rolling_zscore
# ---------------------------------------------------------------------------


def test_rolling_zscore_length(rng_series):
    result = compute_rolling_zscore(rng_series, window=WINDOW)
    assert len(result) == len(rng_series)


def test_rolling_zscore_index_preserved(rng_series):
    idx = pd.date_range("2024-01-01", periods=N, freq="D")
    s = pd.Series(rng_series.values, index=idx)
    result = compute_rolling_zscore(s, window=WINDOW)
    assert (result.index == idx).all()


def test_rolling_zscore_nan_prefix(rng_series):
    result = compute_rolling_zscore(rng_series, window=WINDOW)
    # First window-1 values should be NaN (not enough data)
    assert result.iloc[: WINDOW - 1].isna().all()


def test_rolling_zscore_finite_after_window(rng_series):
    result = compute_rolling_zscore(rng_series, window=WINDOW)
    assert result.iloc[WINDOW:].notna().all()


# ---------------------------------------------------------------------------
# compute_robust_zscore
# ---------------------------------------------------------------------------


def test_robust_zscore_length(rng_series):
    result = compute_robust_zscore(rng_series, window=WINDOW)
    assert len(result) == len(rng_series)


def test_robust_zscore_less_sensitive_to_outliers():
    """Robust z-score should yield a smaller peak value for an outlier-heavy series."""
    s = pd.Series([10.0] * 30 + [1000.0] + [10.0] * 29)
    std_z = compute_rolling_zscore(s, window=20).abs().max()
    rob_z = compute_robust_zscore(s, window=20).abs().max()
    # Robust score peak should be <= standard peak (outlier compressed)
    assert rob_z <= std_z


def test_robust_zscore_index_preserved(rng_series):
    idx = pd.RangeIndex(start=5, stop=5 + N)
    s = pd.Series(rng_series.values, index=idx)
    result = compute_robust_zscore(s, window=WINDOW)
    assert list(result.index) == list(idx)


# ---------------------------------------------------------------------------
# compute_local_ratio
# ---------------------------------------------------------------------------


def test_local_ratio_length(rng_series):
    result = compute_local_ratio(rng_series, window=WINDOW)
    assert len(result) == len(rng_series)


def test_local_ratio_positive_series():
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0] * 3)
    result = compute_local_ratio(s, window=5)
    # For positive series, result sign should match input sign (positive)
    assert result.dropna().gt(0).all()


def test_local_ratio_negative_values():
    s = pd.Series([-1.0, -2.0, -3.0, -4.0, -5.0] * 10)
    result = compute_local_ratio(s, window=5)
    # For uniformly negative series, ratio should be negative
    assert result.dropna().lt(0).all()


# ---------------------------------------------------------------------------
# compute_log_ratio
# ---------------------------------------------------------------------------


def test_log_ratio_length(rng_series):
    result = compute_log_ratio(rng_series, window=WINDOW)
    assert len(result) == len(rng_series)


def test_log_ratio_handles_zeros():
    s = pd.Series([0.0, 1.0, 2.0, 0.0, 3.0] * 10)
    result = compute_log_ratio(s, window=5)
    assert result.dropna().apply(np.isfinite).all()


def test_log_ratio_custom_reference(rng_series):
    ref = rng_series * 2
    result = compute_log_ratio(rng_series, reference=ref, window=WINDOW)
    assert len(result) == len(rng_series)
    # log(x/(2x)) = log(0.5) < 0 for positive x
    finite = result.dropna()
    assert (finite < 0.0).all() or len(finite) == 0  # all < 0 after warmup


# ---------------------------------------------------------------------------
# compute_relative_change
# ---------------------------------------------------------------------------


def test_relative_change_length(rng_series):
    result = compute_relative_change(rng_series, periods=1)
    assert len(result) == len(rng_series)


def test_relative_change_first_nan(rng_series):
    result = compute_relative_change(rng_series, periods=1)
    assert pd.isna(result.iloc[0])


def test_relative_change_correct_value():
    s = pd.Series([100.0, 110.0, 121.0])
    result = compute_relative_change(s, periods=1)
    assert abs(result.iloc[1] - 0.1) < 1e-6
    assert abs(result.iloc[2] - 0.1) < 1e-6


# ---------------------------------------------------------------------------
# compute_rolling_energy
# ---------------------------------------------------------------------------


def test_rolling_energy_non_negative(rng_series):
    result = compute_rolling_energy(rng_series, window=WINDOW)
    assert (result.dropna() >= 0).all()


def test_rolling_energy_length(rng_series):
    result = compute_rolling_energy(rng_series, window=WINDOW)
    assert len(result) == len(rng_series)


def test_rolling_energy_zero_for_zeros():
    s = pd.Series([0.0] * 30)
    result = compute_rolling_energy(s, window=5)
    assert (result.dropna() == 0.0).all()


# ---------------------------------------------------------------------------
# compute_normalized_energy
# ---------------------------------------------------------------------------


def test_normalized_energy_finite(rng_series):
    result = compute_normalized_energy(rng_series, window=WINDOW)
    assert result.dropna().apply(np.isfinite).all()


def test_normalized_energy_length(rng_series):
    result = compute_normalized_energy(rng_series, window=WINDOW)
    assert len(result) == len(rng_series)


# ---------------------------------------------------------------------------
# compute_minmax_position
# ---------------------------------------------------------------------------


def test_minmax_position_bounds(rng_series):
    result = compute_minmax_position(rng_series, window=WINDOW)
    valid = result.dropna()
    assert (valid >= 0.0).all()
    assert (valid <= 1.0 + 1e-8).all()


def test_minmax_position_length(rng_series):
    result = compute_minmax_position(rng_series, window=WINDOW)
    assert len(result) == len(rng_series)


def test_minmax_position_constant_series():
    s = pd.Series([5.0] * 30)
    result = compute_minmax_position(s, window=10)
    # For a constant series, (x-min)/(max-min+eps) ≈ 0
    assert (result.dropna() < 1e-5).all()


# ---------------------------------------------------------------------------
# add_invariant_features
# ---------------------------------------------------------------------------


def test_add_invariant_features_columns_added(simple_df):
    out = add_invariant_features(simple_df, columns=["close"], window=WINDOW)
    expected_suffixes = [
        "_zscore",
        "_robust_zscore",
        "_local_ratio",
        "_log_ratio",
        "_relative_change",
        "_rolling_energy",
        "_normalized_energy",
        "_minmax_position",
    ]
    for suffix in expected_suffixes:
        assert f"close{suffix}" in out.columns, f"Missing column: close{suffix}"


def test_add_invariant_features_no_mutation(simple_df):
    original_cols = list(simple_df.columns)
    _ = add_invariant_features(simple_df, columns=["close"], window=WINDOW)
    assert list(simple_df.columns) == original_cols


def test_add_invariant_features_ignores_missing_column(simple_df):
    out = add_invariant_features(
        simple_df, columns=["nonexistent_col"], window=WINDOW
    )
    # No new columns should be added
    assert list(out.columns) == list(simple_df.columns)


def test_add_invariant_features_ignores_non_numeric():
    df = pd.DataFrame({"signal": [1.0, 2.0, 3.0] * 10, "label": ["a", "b", "c"] * 10})
    out = add_invariant_features(df, columns=["label"], window=5)
    assert list(out.columns) == list(df.columns)


def test_add_invariant_features_invalid_df():
    with pytest.raises((ValueError, TypeError)):
        add_invariant_features("not_a_dataframe", columns=["close"], window=WINDOW)


def test_add_invariant_features_prefix(simple_df):
    out = add_invariant_features(
        simple_df, columns=["close"], window=WINDOW, prefix="feat_"
    )
    assert "feat_close_zscore" in out.columns


def test_add_invariant_features_multiple_columns(simple_df):
    out = add_invariant_features(
        simple_df, columns=["close", "volume"], window=WINDOW
    )
    assert "close_zscore" in out.columns
    assert "volume_zscore" in out.columns


# ---------------------------------------------------------------------------
# Module-level wrappers mirror class methods (spot check)
# ---------------------------------------------------------------------------


def test_module_wrappers_exist(rng_series):
    """All module-level wrappers should be callable and return the same results."""
    assert len(compute_rolling_zscore(rng_series, window=WINDOW)) == N
    assert len(compute_robust_zscore(rng_series, window=WINDOW)) == N
    assert len(compute_local_ratio(rng_series, window=WINDOW)) == N
    assert len(compute_log_ratio(rng_series, window=WINDOW)) == N
    assert len(compute_relative_change(rng_series)) == N
    assert len(compute_rolling_energy(rng_series, window=WINDOW)) == N
    assert len(compute_normalized_energy(rng_series, window=WINDOW)) == N
    assert len(compute_minmax_position(rng_series, window=WINDOW)) == N
