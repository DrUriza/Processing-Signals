"""Tests for signal_analysis.core.sync."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from signal_analysis.core.sync import (
    align_channels,
    align_to_index,
    compute_coverage_ratio,
    compute_missing_ratio,
    diagnose_alignment,
    merge_channels,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def idx_daily() -> pd.DatetimeIndex:
    return pd.date_range("2024-01-01", periods=10, freq="D")


@pytest.fixture
def ref_df(idx_daily) -> pd.DataFrame:
    return pd.DataFrame({"close": np.arange(10.0), "volume": np.arange(10.0) * 100}, index=idx_daily)


@pytest.fixture
def sparse_df(idx_daily) -> pd.DataFrame:
    """DataFrame that covers only every other day."""
    idx_sparse = idx_daily[::2]
    return pd.DataFrame({"value": np.arange(5.0)}, index=idx_sparse)


# ---------------------------------------------------------------------------
# align_to_index
# ---------------------------------------------------------------------------


def test_align_to_index_same_length(ref_df, idx_daily):
    result = align_to_index(ref_df, idx_daily)
    assert len(result) == len(idx_daily)


def test_align_to_index_reindexed_correctly(ref_df, idx_daily):
    new_idx = idx_daily[:5]
    result = align_to_index(ref_df, new_idx)
    assert list(result.index) == list(new_idx)


def test_align_to_index_ffill(sparse_df, idx_daily):
    result = align_to_index(sparse_df, idx_daily, method="ffill")
    # All non-first NaN values should be filled (first row may still be NaN)
    assert result.iloc[1:].notna().all().all()


def test_align_to_index_no_fill(sparse_df, idx_daily):
    result = align_to_index(sparse_df, idx_daily, method=None)
    # Gaps should remain as NaN
    assert result.iloc[1].isna().all()


def test_align_to_index_invalid_df(idx_daily):
    with pytest.raises((ValueError, TypeError)):
        align_to_index("not_a_df", idx_daily)


# ---------------------------------------------------------------------------
# align_channels
# ---------------------------------------------------------------------------


def test_align_channels_all_have_reference_index(ref_df, sparse_df, idx_daily):
    channels = {"ref": ref_df, "sparse": sparse_df}
    aligned = align_channels(channels, reference_key="ref")
    for key, df in aligned.items():
        assert list(df.index) == list(idx_daily), f"{key} index mismatch"


def test_align_channels_missing_ref_key_raises(ref_df):
    with pytest.raises(ValueError):
        align_channels({"ref": ref_df}, reference_key="nonexistent")


def test_align_channels_empty_raises():
    with pytest.raises(ValueError):
        align_channels({}, reference_key="x")


def test_align_channels_single_channel(ref_df, idx_daily):
    channels = {"ref": ref_df}
    aligned = align_channels(channels, reference_key="ref")
    assert list(aligned["ref"].index) == list(idx_daily)


# ---------------------------------------------------------------------------
# compute_coverage_ratio
# ---------------------------------------------------------------------------


def test_coverage_ratio_full_series(idx_daily):
    s = pd.Series(np.arange(10.0), index=idx_daily)
    ratio = compute_coverage_ratio(s, idx_daily)
    assert ratio == 1.0


def test_coverage_ratio_partial(idx_daily):
    # Provide values only for first 5 days
    s = pd.Series(np.arange(5.0), index=idx_daily[:5])
    ratio = compute_coverage_ratio(s, idx_daily)
    assert abs(ratio - 0.5) < 1e-9


def test_coverage_ratio_empty_ref():
    s = pd.Series([1.0, 2.0])
    ratio = compute_coverage_ratio(s, pd.Index([]))
    assert ratio == 0.0


# ---------------------------------------------------------------------------
# compute_missing_ratio
# ---------------------------------------------------------------------------


def test_missing_ratio_no_nans():
    s = pd.Series([1.0, 2.0, 3.0])
    assert compute_missing_ratio(s) == 0.0


def test_missing_ratio_all_nans():
    s = pd.Series([np.nan, np.nan, np.nan])
    assert compute_missing_ratio(s) == 1.0


def test_missing_ratio_partial():
    s = pd.Series([1.0, np.nan, 3.0, np.nan])
    assert abs(compute_missing_ratio(s) - 0.5) < 1e-9


def test_missing_ratio_empty():
    s = pd.Series([], dtype=float)
    assert compute_missing_ratio(s) == 0.0


# ---------------------------------------------------------------------------
# diagnose_alignment
# ---------------------------------------------------------------------------


def test_diagnose_alignment_returns_dataframe(ref_df, sparse_df):
    channels = {"ref": ref_df, "sparse": sparse_df}
    result = diagnose_alignment(channels, reference_key="ref")
    assert isinstance(result, pd.DataFrame)
    assert set(result.columns) == {"coverage_ratio", "missing_ratio", "n_samples"}


def test_diagnose_alignment_index_is_channel_names(ref_df, sparse_df):
    channels = {"ref": ref_df, "sparse": sparse_df}
    result = diagnose_alignment(channels, reference_key="ref")
    assert set(result.index) == {"ref", "sparse"}


def test_diagnose_alignment_ref_has_full_coverage(ref_df):
    channels = {"ref": ref_df}
    result = diagnose_alignment(channels, reference_key="ref")
    assert result.loc["ref", "coverage_ratio"] == 1.0


def test_diagnose_alignment_sparse_has_lower_coverage(ref_df, sparse_df):
    channels = {"ref": ref_df, "sparse": sparse_df}
    result = diagnose_alignment(channels, reference_key="ref")
    assert result.loc["sparse", "coverage_ratio"] < 1.0


def test_diagnose_alignment_empty_raises():
    with pytest.raises(ValueError):
        diagnose_alignment({}, reference_key="x")


def test_diagnose_alignment_missing_ref_key_raises(ref_df):
    with pytest.raises(ValueError):
        diagnose_alignment({"ref": ref_df}, reference_key="nonexistent")


def test_diagnose_alignment_series_channel(ref_df, idx_daily):
    s = pd.Series(np.arange(10.0), index=idx_daily, name="mysignal")
    channels = {"ref": ref_df, "series_ch": s}
    result = diagnose_alignment(channels, reference_key="ref")
    assert "series_ch" in result.index


# ---------------------------------------------------------------------------
# merge_channels
# ---------------------------------------------------------------------------


def test_merge_channels_returns_dataframe(ref_df, sparse_df):
    channels = {"ref": ref_df, "sparse": sparse_df}
    merged = merge_channels(channels, reference_key="ref")
    assert isinstance(merged, pd.DataFrame)


def test_merge_channels_column_prefixes(ref_df, sparse_df):
    channels = {"ref": ref_df, "sparse": sparse_df}
    merged = merge_channels(channels, reference_key="ref")
    for col in merged.columns:
        assert col.startswith("ref_") or col.startswith("sparse_")


def test_merge_channels_row_count(ref_df, sparse_df, idx_daily):
    channels = {"ref": ref_df, "sparse": sparse_df}
    merged = merge_channels(channels, reference_key="ref")
    assert len(merged) == len(idx_daily)


def test_merge_channels_single_channel(ref_df, idx_daily):
    channels = {"ref": ref_df}
    merged = merge_channels(channels, reference_key="ref")
    assert len(merged) == len(idx_daily)
    assert all(c.startswith("ref_") for c in merged.columns)


def test_merge_channels_empty_raises():
    with pytest.raises(ValueError):
        merge_channels({}, reference_key="x")
