import pandas as pd

from signal_analysis.indicators import (
    compute_roc,
    compute_rsi,
    compute_rsi_tsi_regime,
    compute_tsi,
)


def test_compute_rsi_returns_series():
    s = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    out = compute_rsi(s, window=5)
    assert isinstance(out, pd.Series)
    assert out.name == "rsi_5"
    assert len(out) == len(s)


def test_compute_tsi_returns_series():
    s = pd.Series([1, 2, 3, 2, 4, 5, 4, 6, 7, 8])
    out = compute_tsi(s, window_slow=4, window_fast=2)
    assert isinstance(out, pd.Series)
    assert out.name == "tsi_4_2"
    assert len(out) == len(s)


def test_compute_roc_returns_series():
    s = pd.Series([10, 11, 12, 13, 14, 15])
    out = compute_roc(s, window=2)
    assert isinstance(out, pd.Series)
    assert out.name == "roc_2"
    assert len(out) == len(s)


def test_compute_rsi_tsi_regime_returns_dataframe():
    s = pd.Series([1, 2, 3, 2, 4, 5, 4, 6, 7, 8, 7, 9])
    out = compute_rsi_tsi_regime(s, rsi_window=5, tsi_window_slow=4, tsi_window_fast=2)
    assert isinstance(out, pd.DataFrame)
    assert list(out.columns) == ["rsi", "tsi", "rsi_position", "tsi_position"]
    assert len(out) == len(s)
    assert set(out["rsi_position"].dropna().unique()).issubset({-1.0, 0.0, 1.0})
    assert set(out["tsi_position"].dropna().unique()).issubset({-1.0, 0.0, 1.0})


def test_compute_rsi_tsi_regime_invalid_thresholds_raise_value_error():
    s = pd.Series([1, 2, 3, 4, 5, 6, 7])
    try:
        compute_rsi_tsi_regime(s, rsi_low=80.0, rsi_high=20.0)
        assert False, "Expected ValueError for invalid RSI thresholds"
    except ValueError as exc:
        assert "rsi_low" in str(exc)

    try:
        compute_rsi_tsi_regime(s, tsi_long_below=30.0, tsi_short_above=10.0)
        assert False, "Expected ValueError for invalid TSI thresholds"
    except ValueError as exc:
        assert "tsi_long_below" in str(exc)
