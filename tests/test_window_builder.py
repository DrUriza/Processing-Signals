"""Tests for signal_analysis.core.window_builder."""

from __future__ import annotations

import numpy as np
import pandas as pd
import importlib

from signal_analysis.core.feature_window import validate_window
from signal_analysis.core.window_sections import (
    build_technical_indicators_section,
    build_trend_signals_section,
    build_volume_flow_section,
)
from signal_analysis.core.window_builder import (
    WindowBuilder,
    build_feature_sections,
    build_window,
    build_window_sequence,
    flatten_window,
    summarize_frame,
    windows_to_feature_frame,
)


def _make_ohlc_df(n: int = 40) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    base = np.linspace(100.0, 120.0, n)
    noise = rng.normal(0.0, 0.5, n)
    close = base + noise
    open_ = close + rng.normal(0.0, 0.4, n)
    high = np.maximum(open_, close) + np.abs(rng.normal(0.4, 0.1, n))
    low = np.minimum(open_, close) - np.abs(rng.normal(0.4, 0.1, n))
    signal = np.sin(np.linspace(0.0, 4.0, n))
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "signal": signal,
            "category": ["a"] * n,
        }
    )


def test_summarize_frame_expected_stats():
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]})
    out = summarize_frame(df, columns=["x"], prefix="sec_")

    assert out["sec_x_mean"] == 2.5
    assert out["sec_x_min"] == 1.0
    assert out["sec_x_max"] == 4.0
    assert out["sec_x_first"] == 1.0
    assert out["sec_x_last"] == 4.0
    assert out["sec_x_delta"] == 3.0
    assert out["sec_x_median"] == 2.5


def test_summarize_frame_ignores_non_numeric_columns():
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "label": ["a", "b", "c"]})
    out = WindowBuilder.summarize_frame(df)

    assert any(k.startswith("x_") for k in out.keys())
    assert not any(k.startswith("label_") for k in out.keys())


def test_build_feature_sections_populates_variable_action_with_ohlc():
    df = _make_ohlc_df(30)
    sections = build_feature_sections(df)

    assert sections["variable_action"] is not None
    assert isinstance(sections["variable_action"], dict)
    assert len(sections["variable_action"]) > 0


def test_build_feature_sections_populates_invariant_and_dynamics():
    df = _make_ohlc_df(30)
    sections = build_feature_sections(df, value_columns=["close", "signal"])

    assert sections["invariant_features"] is not None
    assert sections["dynamics_features"] is not None
    assert len(sections["invariant_features"]) > 0
    assert len(sections["dynamics_features"]) > 0


def test_build_feature_sections_populates_volume_flow_when_columns_exist():
    df = _make_ohlc_df(30)
    df["volume"] = np.linspace(100.0, 200.0, len(df))
    df["positive_flow"] = np.linspace(1.0, 2.0, len(df))
    sections = build_feature_sections(df)

    assert sections["volume_flow"] is not None
    assert isinstance(sections["volume_flow"], dict)
    assert any(k.startswith("volume_") for k in sections["volume_flow"].keys())


def test_build_volume_flow_section_matches_window_builder_output():
    df = _make_ohlc_df(30)
    df["volume"] = np.linspace(100.0, 200.0, len(df))
    df["positive_flow"] = np.linspace(1.0, 2.0, len(df))

    sections = build_feature_sections(df)
    direct = build_volume_flow_section(df)

    assert sections["volume_flow"] == direct


def test_build_feature_sections_volume_flow_none_when_missing_columns():
    df = _make_ohlc_df(30)
    sections = build_feature_sections(df)

    assert sections["volume_flow"] is None


def test_build_feature_sections_populates_technical_indicators_when_columns_exist():
    df = _make_ohlc_df(30)
    df["rsi"] = np.linspace(30.0, 70.0, len(df))
    df["macd"] = np.linspace(-1.0, 1.0, len(df))
    sections = build_feature_sections(df)

    assert sections["technical_indicators"] is not None
    assert isinstance(sections["technical_indicators"], dict)
    assert any(k.startswith("rsi_") or k.startswith("macd_") for k in sections["technical_indicators"].keys())


def test_build_technical_indicators_section_matches_window_builder_output():
    df = _make_ohlc_df(30)
    df["rsi"] = np.linspace(30.0, 70.0, len(df))
    df["macd"] = np.linspace(-1.0, 1.0, len(df))

    sections = build_feature_sections(df)
    direct = build_technical_indicators_section(df)

    assert sections["technical_indicators"] == direct


def test_build_feature_sections_technical_indicators_none_when_missing_columns():
    df = _make_ohlc_df(30)
    sections = build_feature_sections(df)

    assert sections["technical_indicators"] is None


def test_build_window_has_all_canonical_sections():
    df = _make_ohlc_df(30)
    window = build_window(df, start=0, end=20, window_id="w0")

    expected_keys = {
        "window_meta",
        "variable_action",
        "volume_flow",
        "technical_indicators",
        "trend_signals",
        "fourier_features",
        "wavelet_features",
        "tda_features",
        "invariant_features",
        "dynamics_features",
        "label_or_future_state",
    }
    assert set(window.keys()) == expected_keys


def test_build_window_validates_successfully():
    df = _make_ohlc_df(30)
    window = build_window(df, start=0, end=20)

    validate_window(window)


def test_build_window_populates_volume_flow_and_technical_indicators_when_present():
    df = _make_ohlc_df(30)
    df["volume"] = np.linspace(100.0, 200.0, len(df))
    df["rsi"] = np.linspace(30.0, 70.0, len(df))

    window = build_window(df, start=0, end=20)

    assert window["volume_flow"] is not None
    assert window["technical_indicators"] is not None


def test_window_builder_populates_trend_signals_when_columns_exist():
    df = _make_ohlc_df(35)
    df["rsi"] = np.linspace(30.0, 70.0, len(df))
    df["atr"] = np.linspace(1.0, 2.0, len(df))
    df["volume"] = np.linspace(100.0, 200.0, len(df))

    sections = build_feature_sections(df)

    assert sections["trend_signals"] is not None
    assert isinstance(sections["trend_signals"], dict)
    assert any(k.endswith("_confidence_mean") for k in sections["trend_signals"].keys())


def test_build_trend_signals_section_matches_window_builder_output():
    df = _make_ohlc_df(35)
    df["rsi"] = np.linspace(30.0, 70.0, len(df))
    df["atr"] = np.linspace(1.0, 2.0, len(df))
    df["volume"] = np.linspace(100.0, 200.0, len(df))

    sections = build_feature_sections(df)
    direct = build_trend_signals_section(df)

    assert sections["trend_signals"] == direct


def test_wrapper_module_imports_still_work():
    module_names = [
        "signal_analysis.indicators.momentum",
        "signal_analysis.indicators.moving_averages",
        "signal_analysis.indicators.volatility",
        "signal_analysis.indicators.trend_strength",
        "signal_analysis.indicators.oscillators",
        "signal_analysis.indicators.price_levels",
        "signal_analysis.indicators.volume",
    ]

    for module_name in module_names:
        module = importlib.import_module(module_name)
        assert module is not None


def test_build_window_sequence_expected_count():
    df = _make_ohlc_df(50)
    windows = build_window_sequence(df, window_size=10, step=5)

    expected = ((len(df) - 10) // 5) + 1
    assert len(windows) == expected


def test_flatten_window_skips_none_and_flattens_scalars():
    window = {
        "window_meta": {"window_id": "0", "n_samples": 10},
        "variable_action": {"range_mean": 1.23},
        "volume_flow": None,
        "technical_indicators": None,
        "trend_signals": None,
        "fourier_features": None,
        "wavelet_features": None,
        "tda_features": None,
        "invariant_features": {"close_zscore_mean": 0.2},
        "dynamics_features": {"close_velocity_mean": 0.05},
        "label_or_future_state": None,
    }

    flat = flatten_window(window, include_meta=False)

    assert "variable_action__range_mean" in flat
    assert "invariant_features__close_zscore_mean" in flat
    assert "window_meta__window_id" not in flat
    assert not any(k.startswith("volume_flow") for k in flat)


def test_windows_to_feature_frame_one_row_per_window():
    df = _make_ohlc_df(40)
    windows = build_window_sequence(df, window_size=10, step=10)
    feature_df = windows_to_feature_frame(windows)

    assert isinstance(feature_df, pd.DataFrame)
    assert len(feature_df) == len(windows)


def test_input_dataframe_not_mutated():
    df = _make_ohlc_df(30)
    before = df.copy(deep=True)

    _ = build_feature_sections(df, value_columns=["close", "signal"])
    _ = build_window(df, start=0, end=20)
    _ = build_window_sequence(df, window_size=10, step=5)

    pd.testing.assert_frame_equal(df, before)
