"""Tests for signal_analysis.indicators.dynamics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from signal_analysis.indicators.dynamics import (
    DynamicsIndicators,
    add_dynamics_features,
    compute_acceleration,
    compute_curvature,
    compute_dynamic_profile,
    compute_jerk,
    compute_velocity,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

N = 40


@pytest.fixture
def linear_series() -> pd.Series:
    """Uniformly increasing series: velocity constant, acceleration = 0."""
    return pd.Series(np.arange(N, dtype=float))


@pytest.fixture
def quadratic_series() -> pd.Series:
    """x = t^2: velocity = 2t, acceleration constant = 2."""
    return pd.Series(np.arange(N, dtype=float) ** 2)


@pytest.fixture
def rng_series() -> pd.Series:
    rng = np.random.default_rng(7)
    return pd.Series(rng.normal(50, 5, N))


@pytest.fixture
def simple_df(rng_series) -> pd.DataFrame:
    rng = np.random.default_rng(13)
    return pd.DataFrame(
        {
            "signal": rng_series.values,
            "volume": np.abs(rng.normal(100, 20, N)),
        }
    )


# ---------------------------------------------------------------------------
# compute_velocity
# ---------------------------------------------------------------------------


def test_velocity_length(rng_series):
    result = compute_velocity(rng_series)
    assert len(result) == len(rng_series)


def test_velocity_first_value_nan(rng_series):
    result = compute_velocity(rng_series, periods=1)
    assert pd.isna(result.iloc[0])


def test_velocity_linear_series_is_constant(linear_series):
    result = compute_velocity(linear_series, periods=1)
    valid = result.dropna()
    assert (valid == 1.0).all()


def test_velocity_periods_2(linear_series):
    result = compute_velocity(linear_series, periods=2)
    valid = result.dropna()
    assert (valid == 2.0).all()


def test_velocity_index_preserved(rng_series):
    idx = pd.date_range("2024-01-01", periods=N, freq="D")
    s = pd.Series(rng_series.values, index=idx)
    result = compute_velocity(s)
    assert (result.index == idx).all()


# ---------------------------------------------------------------------------
# compute_acceleration
# ---------------------------------------------------------------------------


def test_acceleration_length(rng_series):
    result = compute_acceleration(rng_series)
    assert len(result) == len(rng_series)


def test_acceleration_linear_series_is_zero(linear_series):
    """Linear series has zero acceleration."""
    result = compute_acceleration(linear_series, periods=1)
    valid = result.dropna()
    assert (valid.abs() < 1e-10).all()


def test_acceleration_quadratic_series_constant(quadratic_series):
    """x=t^2 → a = 2 (constant second difference)."""
    result = compute_acceleration(quadratic_series, periods=1)
    valid = result.dropna()
    assert (np.abs(valid - 2.0) < 1e-8).all()


def test_acceleration_nan_prefix(rng_series):
    result = compute_acceleration(rng_series, periods=1)
    assert result.iloc[:2].isna().all()


# ---------------------------------------------------------------------------
# compute_jerk
# ---------------------------------------------------------------------------


def test_jerk_length(rng_series):
    result = compute_jerk(rng_series)
    assert len(result) == len(rng_series)


def test_jerk_linear_series_is_zero(linear_series):
    """Linear series has zero jerk."""
    result = compute_jerk(linear_series, periods=1)
    valid = result.dropna()
    assert (valid.abs() < 1e-10).all()


def test_jerk_nan_prefix(rng_series):
    result = compute_jerk(rng_series, periods=1)
    assert result.iloc[:3].isna().all()


def test_jerk_quadratic_is_zero(quadratic_series):
    """x=t^2 has constant acceleration → jerk = 0."""
    result = compute_jerk(quadratic_series, periods=1)
    valid = result.dropna()
    assert (valid.abs() < 1e-8).all()


# ---------------------------------------------------------------------------
# compute_curvature
# ---------------------------------------------------------------------------


def test_curvature_length(rng_series):
    result = compute_curvature(rng_series)
    assert len(result) == len(rng_series)


def test_curvature_linear_is_zero(linear_series):
    """Linear series has zero curvature."""
    result = compute_curvature(linear_series)
    valid = result.dropna()
    assert (valid.abs() < 1e-10).all()


def test_curvature_nan_prefix(rng_series):
    result = compute_curvature(rng_series)
    assert result.iloc[:2].isna().all()


def test_curvature_index_preserved(rng_series):
    idx = pd.RangeIndex(start=10, stop=10 + N)
    s = pd.Series(rng_series.values, index=idx)
    result = compute_curvature(s)
    assert list(result.index) == list(idx)


# ---------------------------------------------------------------------------
# compute_dynamic_profile
# ---------------------------------------------------------------------------


def test_dynamic_profile_shape(rng_series):
    result = compute_dynamic_profile(rng_series)
    assert isinstance(result, pd.DataFrame)
    assert result.shape[0] == len(rng_series)
    assert set(result.columns) == {"velocity", "acceleration", "jerk", "curvature"}


def test_dynamic_profile_index_matches(rng_series):
    idx = pd.date_range("2024-01-01", periods=N, freq="D")
    s = pd.Series(rng_series.values, index=idx)
    result = compute_dynamic_profile(s)
    assert (result.index == idx).all()


# ---------------------------------------------------------------------------
# add_dynamics_features
# ---------------------------------------------------------------------------


def test_add_dynamics_columns_added(simple_df):
    out = add_dynamics_features(simple_df, columns=["signal"])
    for suffix in ["_velocity", "_acceleration", "_jerk", "_curvature"]:
        assert f"signal{suffix}" in out.columns, f"Missing: signal{suffix}"


def test_add_dynamics_no_mutation(simple_df):
    original_cols = list(simple_df.columns)
    _ = add_dynamics_features(simple_df, columns=["signal"])
    assert list(simple_df.columns) == original_cols


def test_add_dynamics_ignores_missing_column(simple_df):
    out = add_dynamics_features(simple_df, columns=["nonexistent"])
    assert list(out.columns) == list(simple_df.columns)


def test_add_dynamics_ignores_non_numeric():
    df = pd.DataFrame({"x": [1.0, 2.0] * 10, "label": ["a", "b"] * 10})
    out = add_dynamics_features(df, columns=["label"])
    assert list(out.columns) == list(df.columns)


def test_add_dynamics_multiple_columns(simple_df):
    out = add_dynamics_features(simple_df, columns=["signal", "volume"])
    assert "signal_velocity" in out.columns
    assert "volume_velocity" in out.columns


def test_add_dynamics_prefix(simple_df):
    out = add_dynamics_features(simple_df, columns=["signal"], prefix="dyn_")
    assert "dyn_signal_velocity" in out.columns


def test_add_dynamics_invalid_df():
    with pytest.raises((ValueError, TypeError)):
        add_dynamics_features("not_a_df", columns=["x"])


# ---------------------------------------------------------------------------
# Module-level wrappers (spot check)
# ---------------------------------------------------------------------------


def test_module_wrappers(rng_series):
    assert len(compute_velocity(rng_series)) == N
    assert len(compute_acceleration(rng_series)) == N
    assert len(compute_jerk(rng_series)) == N
    assert len(compute_curvature(rng_series)) == N
    profile = compute_dynamic_profile(rng_series)
    assert len(profile) == N
