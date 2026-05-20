"""Tests for signal_analysis.core.feature_pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from signal_analysis.core.feature_pipeline import (
    build_feature_matrix,
    build_pipeline_report,
    drop_non_numeric_features,
    fill_feature_matrix,
    validate_feature_matrix,
)


def _make_df(n: int = 40) -> pd.DataFrame:
    rng = np.random.default_rng(123)
    base = np.linspace(100.0, 120.0, n)
    close = base + rng.normal(0.0, 0.4, n)
    open_ = close + rng.normal(0.0, 0.3, n)
    high = np.maximum(open_, close) + np.abs(rng.normal(0.4, 0.1, n))
    low = np.minimum(open_, close) - np.abs(rng.normal(0.4, 0.1, n))
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "signal": np.sin(np.linspace(0.0, 4.0, n)),
            "label": ["x"] * n,
        }
    )


def test_build_feature_matrix_rows_match_windows():
    df = _make_df(40)
    X = build_feature_matrix(df, window_size=10, step=5)
    expected = ((len(df) - 10) // 5) + 1
    assert len(X) == expected


def test_build_feature_matrix_no_mutation():
    df = _make_df(35)
    before = df.copy(deep=True)
    _ = build_feature_matrix(df, window_size=10, step=5, value_columns=["close", "signal"])
    pd.testing.assert_frame_equal(df, before)


def test_build_feature_matrix_drop_non_numeric_true():
    df = _make_df(30)
    X = build_feature_matrix(df, window_size=10, step=5, include_meta=False, drop_non_numeric=True)
    assert all(pd.api.types.is_numeric_dtype(X[c]) for c in X.columns)


def test_fill_feature_matrix_methods():
    X = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [np.nan, 2.0, np.nan]})

    out_none = fill_feature_matrix(X, method=None)
    assert out_none.isna().sum().sum() == X.isna().sum().sum()

    out_zero = fill_feature_matrix(X, method="zero")
    assert out_zero.isna().sum().sum() == 0
    assert (out_zero == 0).any().any()

    out_ffill = fill_feature_matrix(X, method="ffill")
    assert out_ffill.isna().sum().sum() == 0

    out_median = fill_feature_matrix(X, method="median")
    assert out_median.isna().sum().sum() == 0


def test_validate_feature_matrix_valid_numeric_passes():
    X = pd.DataFrame({"f1": [1.0, 2.0], "f2": [3.0, 4.0]})
    validate_feature_matrix(X, require_numeric=True, allow_nan=True)


def test_validate_feature_matrix_raises_on_non_numeric_when_required():
    X = pd.DataFrame({"f1": [1.0, 2.0], "name": ["a", "b"]})
    with pytest.raises(ValueError):
        validate_feature_matrix(X, require_numeric=True, allow_nan=True)


def test_validate_feature_matrix_raises_on_nan_when_not_allowed():
    X = pd.DataFrame({"f1": [1.0, np.nan], "f2": [2.0, 3.0]})
    with pytest.raises(ValueError):
        validate_feature_matrix(X, require_numeric=True, allow_nan=False)


def test_drop_non_numeric_features_removes_object_columns():
    X = pd.DataFrame({"f1": [1.0, 2.0], "text": ["a", "b"], "f2": [3, 4]})
    out = drop_non_numeric_features(X)
    assert list(out.columns) == ["f1", "f2"]


def test_build_pipeline_report_expected_keys():
    X = pd.DataFrame({"f1": [1.0, np.nan], "f2": [2.0, 2.0], "name": ["a", "b"]})
    report = build_pipeline_report(X)

    expected = {
        "n_rows",
        "n_features",
        "missing_ratio",
        "non_numeric_columns",
        "constant_columns",
    }
    assert set(report.keys()) == expected


def test_build_pipeline_report_detects_constant_columns():
    X = pd.DataFrame({"a": [1.0, 1.0, 1.0], "b": [1.0, 2.0, 3.0], "c": ["x", "x", "x"]})
    report = build_pipeline_report(X)
    assert "a" in report["constant_columns"]
    assert "c" in report["constant_columns"]


def test_build_feature_matrix_includes_volume_flow_and_technical_sections():
    df = _make_df(40)
    df["volume"] = np.linspace(100.0, 200.0, len(df))
    df["rsi"] = np.linspace(30.0, 70.0, len(df))

    X = build_feature_matrix(df, window_size=10, step=5)

    assert any(c.startswith("volume_flow__") for c in X.columns)
    assert any(c.startswith("technical_indicators__") for c in X.columns)


def test_build_feature_matrix_includes_trend_signals_confidence_features():
    df = _make_df(40)
    df["rsi"] = np.linspace(30.0, 70.0, len(df))
    df["atr"] = np.linspace(1.0, 2.0, len(df))
    df["volume"] = np.linspace(100.0, 200.0, len(df))

    X = build_feature_matrix(df, window_size=10, step=5)

    assert any(c.startswith("trend_signals__") for c in X.columns)
    assert any("confidence" in c for c in X.columns if c.startswith("trend_signals__"))
