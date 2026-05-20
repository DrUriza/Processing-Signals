"""Tests for signal_analysis.indicators.states."""

from __future__ import annotations

import numpy as np
import pandas as pd

from signal_analysis.indicators.states import (
    build_signal_state_frame,
    compute_activity_state,
    compute_momentum_state,
    compute_slope_state,
    compute_structure_state,
    compute_trend_state,
    compute_volatility_state,
)


def _make_base_df(n: int = 40) -> pd.DataFrame:
    x = np.linspace(100.0, 120.0, n)
    close = pd.Series(x + np.sin(np.linspace(0, 3, n)) * 0.2)
    open_ = close + 0.1
    high = close + 0.5
    low = close - 0.5
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close})


def test_compute_slope_state_detects_rising_falling_flat():
    rising = pd.Series(np.arange(30, dtype=float))
    falling = pd.Series(np.arange(30, 0, -1, dtype=float))
    flat = pd.Series(np.ones(30, dtype=float) * 10.0)

    r = compute_slope_state(rising, window=5)
    f = compute_slope_state(falling, window=5)
    z = compute_slope_state(flat, window=5)

    assert r["slope_state"].iloc[-1] == "rising"
    assert f["slope_state"].iloc[-1] == "falling"
    assert z["slope_state"].iloc[-1] == "flat"


def test_compute_trend_state_returns_expected_states():
    n = 30
    df_rising = pd.DataFrame(
        {
            "close": np.linspace(1, 10, n),
            "fast": np.linspace(1, 10, n) + 0.5,
            "slow": np.linspace(1, 10, n),
        }
    )
    df_falling = pd.DataFrame(
        {
            "close": np.linspace(10, 1, n),
            "fast": np.linspace(10, 1, n) - 0.5,
            "slow": np.linspace(10, 1, n),
        }
    )
    df_sideways = pd.DataFrame({"close": np.ones(n) * 5.0})

    tr = compute_trend_state(df_rising, value_col="close", fast_col="fast", slow_col="slow")
    tf = compute_trend_state(df_falling, value_col="close", fast_col="fast", slow_col="slow")
    ts = compute_trend_state(df_sideways, value_col="close")

    assert tr["trend_state"].iloc[-1] == "rising"
    assert tf["trend_state"].iloc[-1] == "falling"
    assert ts["trend_state"].iloc[-1] == "sideways"


def test_compute_momentum_state_missing_columns_returns_neutral():
    df = _make_base_df(30)
    out = compute_momentum_state(df, momentum_cols=["not_present_1", "not_present_2"])

    assert (out["momentum_state"] == "neutral").all()
    assert (out["momentum_confidence"] == 0.0).all()


def test_compute_volatility_state_detects_compression_and_expansion():
    n = 60
    atr = np.concatenate(
        [
            np.ones(20) * 10.0,
            np.ones(20) * 2.0,
            np.ones(20) * 25.0,
        ]
    )
    df = pd.DataFrame({"atr": atr})
    out = compute_volatility_state(df, window=10)

    states = set(out["volatility_state"].dropna().unique().tolist())
    assert "compression" in states
    assert "expansion" in states


def test_compute_structure_state_with_existing_geometry_columns():
    df = pd.DataFrame(
        {
            "body_ratio": [0.7, 0.05, 0.3],
            "upper_wick_ratio": [0.1, 0.5, 0.6],
            "lower_wick_ratio": [0.1, 0.6, 0.1],
            "body_direction": [1.0, 0.0, -1.0],
        }
    )
    out = compute_structure_state(df)
    assert "structure_state" in out.columns
    assert "structure_confidence" in out.columns


def test_compute_structure_state_from_ohlc_when_geometry_missing():
    df = _make_base_df(30)
    out = compute_structure_state(df)

    assert "structure_state" in out.columns
    assert "structure_confidence" in out.columns
    assert len(out) == len(df)


def test_compute_activity_state_returns_normal_when_no_activity_columns():
    df = _make_base_df(25)
    out = compute_activity_state(df, activity_cols=["missing_activity_col"])

    assert (out["activity_state"] == "normal").all()
    assert (out["activity_confidence"] == 0.0).all()


def test_build_signal_state_frame_expected_columns():
    df = _make_base_df(40)
    df["rsi"] = np.linspace(30, 70, len(df))
    df["atr"] = np.linspace(1, 2, len(df))
    df["volume"] = np.linspace(100, 200, len(df))

    out = build_signal_state_frame(df)

    expected = {
        "trend_state",
        "trend_confidence",
        "momentum_state",
        "momentum_confidence",
        "volatility_state",
        "volatility_confidence",
        "structure_state",
        "structure_confidence",
        "activity_state",
        "activity_confidence",
    }
    assert expected.issubset(set(out.columns))


def test_states_input_dataframe_not_mutated():
    df = _make_base_df(40)
    df["rsi"] = np.linspace(30, 70, len(df))
    before = df.copy(deep=True)

    _ = compute_trend_state(df)
    _ = compute_momentum_state(df)
    _ = compute_volatility_state(df)
    _ = compute_structure_state(df)
    _ = compute_activity_state(df)
    _ = build_signal_state_frame(df)

    pd.testing.assert_frame_equal(df, before)
