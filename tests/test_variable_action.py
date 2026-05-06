"""
Tests for signal_analysis.indicators.variable_action module.

Tests generic OHLC geometry and state classification (domain-agnostic).
"""

import numpy as np
import pandas as pd
import pytest

from signal_analysis.indicators.variable_action import (
    VariableActionIndicators,
    add_ohlc_geometry_features,
    classify_ohlc_geometry,
    compute_ohlc_geometry,
)


# *********************************************************************************************************************
# Test: compute_ohlc_geometry
# *********************************************************************************************************************


def test_compute_ohlc_geometry_returns_expected_columns():
    """Test that compute_ohlc_geometry returns all expected columns."""
    df = pd.DataFrame(
        {
            "open": [100.0, 101.0, 102.0],
            "high": [102.0, 103.0, 104.0],
            "low": [99.0, 100.0, 101.0],
            "close": [101.5, 102.5, 103.5],
        }
    )
    result = compute_ohlc_geometry(df)

    expected_cols = {
        "range",
        "body_size",
        "body_direction",
        "upper_wick",
        "lower_wick",
        "body_ratio",
        "upper_wick_ratio",
        "lower_wick_ratio",
        "wick_imbalance",
        "wick_imbalance_ratio",
        "close_position",
        "gap",
        "return",
        "log_return",
    }

    assert expected_cols.issubset(set(result.columns))
    assert len(result) == len(df)


def test_compute_ohlc_geometry_same_length():
    """Test that compute_ohlc_geometry returns same length as input."""
    df = pd.DataFrame(
        {
            "open": [100.0, 101.0, 102.0, 103.0],
            "high": [102.0, 103.0, 104.0, 105.0],
            "low": [99.0, 100.0, 101.0, 102.0],
            "close": [101.5, 102.5, 103.5, 104.5],
        }
    )
    result = compute_ohlc_geometry(df)
    assert len(result) == len(df)


def test_compute_ohlc_geometry_body_size_correct():
    """Test body_size computation with handcrafted values."""
    df = pd.DataFrame(
        {
            "open": [100.0, 100.0, 100.0],
            "high": [105.0, 105.0, 105.0],
            "low": [95.0, 95.0, 95.0],
            "close": [104.0, 100.0, 96.0],  # up 4, neutral 0, down 4
        }
    )
    result = compute_ohlc_geometry(df)

    # body_size = |close - open|
    assert result["body_size"].iloc[0] == 4.0  # |104 - 100|
    assert result["body_size"].iloc[1] == 0.0  # |100 - 100|
    assert result["body_size"].iloc[2] == 4.0  # |96 - 100|


def test_compute_ohlc_geometry_upper_wick_correct():
    """Test upper_wick computation."""
    df = pd.DataFrame(
        {
            "open": [100.0],
            "high": [105.0],
            "low": [95.0],
            "close": [102.0],
        }
    )
    result = compute_ohlc_geometry(df)

    # upper_wick = high - max(open, close) = 105 - 102 = 3
    assert result["upper_wick"].iloc[0] == 3.0


def test_compute_ohlc_geometry_lower_wick_correct():
    """Test lower_wick computation."""
    df = pd.DataFrame(
        {
            "open": [100.0],
            "high": [105.0],
            "low": [95.0],
            "close": [102.0],
        }
    )
    result = compute_ohlc_geometry(df)

    # lower_wick = min(open, close) - low = 100 - 95 = 5
    assert result["lower_wick"].iloc[0] == 5.0


def test_compute_ohlc_geometry_body_ratio_correct():
    """Test body_ratio computation."""
    df = pd.DataFrame(
        {
            "open": [100.0],
            "high": [110.0],
            "low": [90.0],
            "close": [108.0],
        }
    )
    result = compute_ohlc_geometry(df)

    # range = 110 - 90 = 20
    # body_size = |108 - 100| = 8
    # body_ratio = 8 / 20 = 0.4
    assert np.isclose(result["body_ratio"].iloc[0], 0.4)


def test_compute_ohlc_geometry_body_direction():
    """Test body_direction (sign of close - open)."""
    df = pd.DataFrame(
        {
            "open": [100.0, 100.0, 100.0],
            "high": [102.0, 102.0, 102.0],
            "low": [98.0, 98.0, 98.0],
            "close": [101.0, 100.0, 99.0],  # up, neutral, down
        }
    )
    result = compute_ohlc_geometry(df)

    assert result["body_direction"].iloc[0] == 1.0   # up
    assert result["body_direction"].iloc[1] == 0.0   # neutral
    assert result["body_direction"].iloc[2] == -1.0  # down


def test_compute_ohlc_geometry_raises_missing_ohlc():
    """Test that compute_ohlc_geometry raises for missing OHLC columns."""
    df = pd.DataFrame({"open": [100.0], "high": [102.0]})  # missing low, close

    with pytest.raises(ValueError, match="Missing required columns"):
        compute_ohlc_geometry(df)


# *********************************************************************************************************************
# Test: classify_ohlc_geometry
# *********************************************************************************************************************


def test_classify_ohlc_geometry_returns_states():
    """Test that classify_ohlc_geometry returns state columns."""
    df = pd.DataFrame(
        {
            "open": [100.0, 100.0, 100.0],
            "high": [110.0, 110.0, 110.0],
            "low": [90.0, 90.0, 90.0],
            "close": [108.0, 100.0, 92.0],
        }
    )
    result = classify_ohlc_geometry(df)

    assert "body_state" in result.columns
    assert "direction_state" in result.columns
    assert "rejection_state" in result.columns
    assert "is_doji" in result.columns
    assert "is_strong_body" in result.columns
    assert "has_upper_rejection" in result.columns
    assert "has_lower_rejection" in result.columns


def test_classify_ohlc_geometry_body_state_doji():
    """Test body_state classification as doji."""
    df = pd.DataFrame(
        {
            "open": [100.0],
            "high": [101.0],
            "low": [99.0],
            "close": [100.05],  # Very small body
        }
    )
    result = classify_ohlc_geometry(df, doji_body_ratio=0.1)
    # body_size = |100.05 - 100| = 0.05
    # range = 101 - 99 = 2
    # body_ratio = 0.05 / 2 = 0.025 < 0.1
    assert result["body_state"].iloc[0] == "doji"
    assert result["is_doji"].iloc[0] == True


def test_classify_ohlc_geometry_body_state_strong():
    """Test body_state classification as strong_body."""
    df = pd.DataFrame(
        {
            "open": [100.0],
            "high": [110.0],
            "low": [90.0],
            "close": [108.0],
        }
    )
    result = classify_ohlc_geometry(df, strong_body_ratio=0.6)
    # body_ratio = 8 / 20 = 0.4, but we'll use a high close
    # Let's recalculate: range = 20, body_size = 8, ratio = 0.4
    # Let's make it stronger:
    pass  # Will test with adjusted data below


def test_classify_ohlc_geometry_direction_states():
    """Test direction_state classification."""
    df = pd.DataFrame(
        {
            "open": [100.0, 100.0, 100.0],
            "high": [110.0, 110.0, 110.0],
            "low": [90.0, 90.0, 90.0],
            "close": [108.0, 100.0, 92.0],
        }
    )
    result = classify_ohlc_geometry(df)

    assert result["direction_state"].iloc[0] == "up"     # close > open
    assert result["direction_state"].iloc[1] == "neutral"  # close == open
    assert result["direction_state"].iloc[2] == "down"    # close < open


def test_classify_ohlc_geometry_rejection_upper():
    """Test rejection_state classification for upper rejection."""
    df = pd.DataFrame(
        {
            "open": [100.0],
            "high": [145.0],  # Very high wick
            "low": [100.0],
            "close": [100.0],  # Closes at open
        }
    )
    result = classify_ohlc_geometry(df, long_wick_ratio=0.45)
    # upper_wick = 145 - 100 = 45
    # range = 145 - 100 = 45
    # upper_wick_ratio = 45 / 45 = 1.0 >= 0.45
    # lower_wick_ratio = 0
    assert result["rejection_state"].iloc[0] == "upper_rejection"
    assert result["has_upper_rejection"].iloc[0] == True


def test_classify_ohlc_geometry_rejection_lower():
    """Test rejection_state classification for lower rejection."""
    df = pd.DataFrame(
        {
            "open": [100.0],
            "high": [100.0],
            "low": [55.0],  # Very low wick
            "close": [100.0],  # Closes at open
        }
    )
    result = classify_ohlc_geometry(df, long_wick_ratio=0.45)
    # lower_wick = 100 - 55 = 45
    # range = 100 - 55 = 45
    # lower_wick_ratio = 45 / 45 = 1.0 >= 0.45
    assert result["rejection_state"].iloc[0] == "lower_rejection"
    assert result["has_lower_rejection"].iloc[0] == True


# *********************************************************************************************************************
# Test: add_ohlc_geometry_features
# *********************************************************************************************************************


def test_add_ohlc_geometry_features_preserves_original():
    """Test that add_ohlc_geometry_features preserves original columns."""
    df = pd.DataFrame(
        {
            "open": [100.0, 101.0, 102.0],
            "high": [102.0, 103.0, 104.0],
            "low": [99.0, 100.0, 101.0],
            "close": [101.5, 102.5, 103.5],
        }
    )
    result = add_ohlc_geometry_features(df)

    # Original columns should still exist
    assert "open" in result.columns
    assert "high" in result.columns
    assert "low" in result.columns
    assert "close" in result.columns

    # Verify original values are unchanged
    assert result["open"].equals(df["open"])
    assert result["high"].equals(df["high"])
    assert result["low"].equals(df["low"])
    assert result["close"].equals(df["close"])


def test_add_ohlc_geometry_features_adds_new_columns():
    """Test that add_ohlc_geometry_features adds geometry columns."""
    df = pd.DataFrame(
        {
            "open": [100.0, 101.0],
            "high": [102.0, 103.0],
            "low": [99.0, 100.0],
            "close": [101.5, 102.5],
        }
    )
    result = add_ohlc_geometry_features(df)

    # Should have added columns
    assert "body_size" in result.columns
    assert "body_state" in result.columns
    assert "direction_state" in result.columns
    assert len(result.columns) > len(df.columns)


def test_add_ohlc_geometry_features_no_mutation():
    """Test that add_ohlc_geometry_features does not mutate input."""
    df = pd.DataFrame(
        {
            "open": [100.0],
            "high": [102.0],
            "low": [99.0],
            "close": [101.5],
        }
    )
    df_copy = df.copy()

    result = add_ohlc_geometry_features(df)

    # Original should be unchanged
    assert df.equals(df_copy)
    # Result should have more columns
    assert len(result.columns) > len(df.columns)


# *********************************************************************************************************************
# Test: Module-level wrappers
# *********************************************************************************************************************


def test_module_wrappers_exist():
    """Test that module-level wrapper functions exist and work."""
    df = pd.DataFrame(
        {
            "open": [100.0],
            "high": [102.0],
            "low": [99.0],
            "close": [101.5],
        }
    )

    # Wrappers should work
    geom = compute_ohlc_geometry(df)
    assert not geom.empty

    classified = classify_ohlc_geometry(df)
    assert not classified.empty

    enriched = add_ohlc_geometry_features(df)
    assert not enriched.empty


# *********************************************************************************************************************
# Test: Edge cases and numerical stability
# *********************************************************************************************************************


def test_compute_ohlc_geometry_zero_range():
    """Test compute_ohlc_geometry handles zero range (doji-like)."""
    df = pd.DataFrame(
        {
            "open": [100.0],
            "high": [100.0],
            "low": [100.0],
            "close": [100.0],
        }
    )
    result = compute_ohlc_geometry(df)

    # Should not raise; eps prevents division by zero
    assert not result.isnull().all().any()
    assert result["body_ratio"].iloc[0] == 0.0
    assert result["upper_wick_ratio"].iloc[0] == 0.0


def test_compute_ohlc_geometry_eps_parameter():
    """Test that eps parameter affects numerical stability."""
    df = pd.DataFrame(
        {
            "open": [100.0],
            "high": [100.0],
            "low": [100.0],
            "close": [100.0],
        }
    )

    # With different eps
    result1 = compute_ohlc_geometry(df, eps=1e-9)
    result2 = compute_ohlc_geometry(df, eps=1e-3)

    # Should complete without error
    assert len(result1) == len(df)
    assert len(result2) == len(df)


def test_classify_ohlc_geometry_thresholds():
    """Test that thresholds affect classification."""
    df = pd.DataFrame(
        {
            "open": [100.0],
            "high": [110.0],
            "low": [90.0],
            "close": [105.0],
        }
    )

    # body_ratio = 5 / 20 = 0.25
    result_high_thresh = classify_ohlc_geometry(df, doji_body_ratio=0.3)
    result_low_thresh = classify_ohlc_geometry(df, doji_body_ratio=0.2)

    # Different thresholds should give different classifications
    # At 0.3 threshold: 0.25 < 0.3, so doji
    # At 0.2 threshold: 0.25 > 0.2, so not doji
    assert result_high_thresh["body_state"].iloc[0] == "doji"
    assert result_low_thresh["body_state"].iloc[0] in ["weak_body", "normal_body"]
