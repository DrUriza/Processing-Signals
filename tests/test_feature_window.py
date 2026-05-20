"""Tests for signal_analysis.core.feature_window."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from signal_analysis import validate_window as top_level_validate_window
from signal_analysis.core.feature_window import (
    WINDOW_META_KEYS,
    WINDOW_SECTION_KEYS,
    build_empty_window,
    build_window_meta,
    build_windowed_sequence,
    extract_window_from_dataframe,
    validate_window,
)
from signal_analysis.utils import validate_positive_window


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def ohlc_df() -> pd.DataFrame:
    n = 100
    rng = np.random.default_rng(0)
    close = rng.uniform(90, 110, n).cumsum() + 100
    high = close + rng.uniform(0, 5, n)
    low = close - rng.uniform(0, 5, n)
    open_ = close - rng.uniform(-3, 3, n)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close})


@pytest.fixture
def ohlc_df_with_time(ohlc_df) -> pd.DataFrame:
    df = ohlc_df.copy()
    df["timestamp"] = pd.date_range("2024-01-01", periods=len(df), freq="h")
    return df


# ---------------------------------------------------------------------------
# WINDOW_SECTION_KEYS and WINDOW_META_KEYS
# ---------------------------------------------------------------------------


def test_window_section_keys_non_empty():
    assert len(WINDOW_SECTION_KEYS) > 0


def test_window_meta_keys_non_empty():
    assert len(WINDOW_META_KEYS) > 0


def test_required_meta_keys_present():
    for key in ["window_id", "n_samples", "schema_version"]:
        assert key in WINDOW_META_KEYS


# ---------------------------------------------------------------------------
# build_window_meta
# ---------------------------------------------------------------------------


def test_build_window_meta_contains_required_keys():
    meta = build_window_meta(window_id=0, n_samples=20)
    for key in WINDOW_META_KEYS:
        assert key in meta


def test_build_window_meta_schema_version():
    meta = build_window_meta(window_id=0, n_samples=20, schema_version="2.0")
    assert meta["schema_version"] == "2.0"


def test_build_window_meta_extra_keys():
    meta = build_window_meta(window_id=0, n_samples=20, extra={"source": "radar"})
    assert meta["source"] == "radar"


# ---------------------------------------------------------------------------
# build_empty_window
# ---------------------------------------------------------------------------


def test_build_empty_window_has_all_sections():
    w = build_empty_window()
    for key in WINDOW_SECTION_KEYS:
        assert key in w


def test_build_empty_window_all_none():
    w = build_empty_window()
    assert all(v is None for v in w.values())


# ---------------------------------------------------------------------------
# extract_window_from_dataframe
# ---------------------------------------------------------------------------


def test_extract_window_returns_dict(ohlc_df):
    w = extract_window_from_dataframe(ohlc_df, start=0, end=20)
    assert isinstance(w, dict)


def test_extract_window_has_all_sections(ohlc_df):
    w = extract_window_from_dataframe(ohlc_df, start=0, end=20)
    for key in WINDOW_SECTION_KEYS:
        assert key in w


def test_extract_window_meta_n_samples(ohlc_df):
    w = extract_window_from_dataframe(ohlc_df, start=5, end=25)
    assert w["window_meta"]["n_samples"] == 20


def test_extract_window_variable_action_shape(ohlc_df):
    w = extract_window_from_dataframe(ohlc_df, start=0, end=30)
    va = w["variable_action"]
    assert isinstance(va, pd.DataFrame)
    assert len(va) == 30
    assert set(["open", "high", "low", "close"]).issubset(set(va.columns))


def test_extract_window_no_mutation(ohlc_df):
    original_len = len(ohlc_df)
    _ = extract_window_from_dataframe(ohlc_df, start=0, end=20)
    assert len(ohlc_df) == original_len


def test_extract_window_with_time_column(ohlc_df_with_time):
    w = extract_window_from_dataframe(ohlc_df_with_time, start=0, end=10)
    meta = w["window_meta"]
    assert meta["time_column"] == "timestamp"
    assert meta["time_start"] is not None
    assert meta["time_end"] is not None


def test_extract_window_custom_id(ohlc_df):
    w = extract_window_from_dataframe(ohlc_df, start=0, end=10, window_id="win_A")
    assert w["window_meta"]["window_id"] == "win_A"


def test_extract_window_missing_ohlc_raises():
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError):
        extract_window_from_dataframe(df, start=0, end=2)


# ---------------------------------------------------------------------------
# build_windowed_sequence
# ---------------------------------------------------------------------------


def test_build_windowed_sequence_count(ohlc_df):
    windows = build_windowed_sequence(ohlc_df, window_size=20, step=20)
    expected = len(ohlc_df) // 20
    assert len(windows) == expected


def test_build_windowed_sequence_overlapping(ohlc_df):
    windows = build_windowed_sequence(ohlc_df, window_size=20, step=10)
    assert len(windows) > len(ohlc_df) // 20


def test_build_windowed_sequence_each_is_dict(ohlc_df):
    windows = build_windowed_sequence(ohlc_df, window_size=10)
    for w in windows:
        assert isinstance(w, dict)


def test_build_windowed_sequence_window_size_correct(ohlc_df):
    windows = build_windowed_sequence(ohlc_df, window_size=15)
    for w in windows:
        assert w["window_meta"]["n_samples"] == 15


def test_build_windowed_sequence_invalid_window_size(ohlc_df):
    with pytest.raises(ValueError):
        build_windowed_sequence(ohlc_df, window_size=0)


def test_build_windowed_sequence_invalid_step(ohlc_df):
    with pytest.raises(ValueError):
        build_windowed_sequence(ohlc_df, window_size=10, step=0)


def test_build_windowed_sequence_ids_sequential(ohlc_df):
    windows = build_windowed_sequence(ohlc_df, window_size=10)
    ids = [w["window_meta"]["window_id"] for w in windows]
    assert ids == list(range(len(windows)))


# ---------------------------------------------------------------------------
# validate_window
# ---------------------------------------------------------------------------


def test_validate_window_valid(ohlc_df):
    w = extract_window_from_dataframe(ohlc_df, start=0, end=10)
    validate_window(w)  # should not raise


def test_validate_window_empty_dict_raises():
    with pytest.raises(ValueError):
        validate_window({})


def test_validate_window_not_dict_raises():
    with pytest.raises(ValueError):
        validate_window("not_a_dict")


def test_validate_window_missing_section_raises():
    w = build_empty_window()
    del w[WINDOW_SECTION_KEYS[0]]
    with pytest.raises(ValueError):
        validate_window(w)


def test_validate_positive_window_accepts_positive_integer():
    validate_positive_window(5)


def test_validate_positive_window_rejects_non_positive_integer():
    with pytest.raises(ValueError):
        validate_positive_window(0)


def test_top_level_validate_window_validates_canonical_feature_window(ohlc_df):
    w = extract_window_from_dataframe(ohlc_df, start=0, end=10)
    top_level_validate_window(w)
