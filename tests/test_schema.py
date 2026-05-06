"""
Tests for signal_analysis.core.schema module.

Test the domain-agnostic validation contract and feature schema functions.
"""

import pandas as pd
import pytest

from signal_analysis.core.schema import (
    OPTIONAL_SIGNED_FLOW_COLUMNS,
    OPTIONAL_TIME_COLUMNS,
    OPTIONAL_VOLUME_COLUMNS,
    REQUIRED_OHLC_COLUMNS,
    build_feature_contract,
    infer_time_column,
    validate_dataframe,
    validate_numeric_columns,
    validate_ohlc_columns,
    validate_required_columns,
)


# *********************************************************************************************************************
# Test: validate_dataframe
# *********************************************************************************************************************


def test_validate_dataframe_accepts_non_empty_dataframe():
    """Test that validate_dataframe accepts a non-empty DataFrame."""
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    # Should not raise
    validate_dataframe(df)


def test_validate_dataframe_raises_for_non_dataframe():
    """Test that validate_dataframe raises ValueError for non-DataFrame input."""
    with pytest.raises(ValueError, match="Input must be a pandas DataFrame"):
        validate_dataframe([1, 2, 3])

    with pytest.raises(ValueError, match="Input must be a pandas DataFrame"):
        validate_dataframe({"a": [1, 2, 3]})

    with pytest.raises(ValueError, match="Input must be a pandas DataFrame"):
        validate_dataframe("not a dataframe")


def test_validate_dataframe_raises_for_empty_dataframe():
    """Test that validate_dataframe raises ValueError for empty DataFrame."""
    df_empty = pd.DataFrame()
    with pytest.raises(ValueError, match="DataFrame is empty"):
        validate_dataframe(df_empty)


# *********************************************************************************************************************
# Test: validate_required_columns
# *********************************************************************************************************************


def test_validate_required_columns_passes_when_present():
    """Test that validate_required_columns passes when all columns exist."""
    df = pd.DataFrame({"open": [1, 2], "high": [3, 4], "low": [0.5, 1.5]})
    # Should not raise
    validate_required_columns(df, ["open", "high", "low"])


def test_validate_required_columns_raises_for_missing_columns():
    """Test that validate_required_columns raises ValueError and lists missing columns."""
    df = pd.DataFrame({"open": [1, 2], "high": [3, 4]})
    with pytest.raises(ValueError, match="Missing required columns"):
        validate_required_columns(df, ["open", "high", "low", "close"])

    # Verify missing columns are listed
    try:
        validate_required_columns(df, ["open", "high", "low", "close"])
    except ValueError as e:
        error_msg = str(e)
        assert "low" in error_msg
        assert "close" in error_msg


# *********************************************************************************************************************
# Test: validate_numeric_columns
# *********************************************************************************************************************


def test_validate_numeric_columns_ignores_missing_columns():
    """Test that validate_numeric_columns ignores columns not present in df."""
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
    # Should not raise, even though 'c' and 'd' don't exist
    validate_numeric_columns(df, ["a", "b", "c", "d"])


def test_validate_numeric_columns_raises_for_non_numeric():
    """Test that validate_numeric_columns raises ValueError for non-numeric existing columns."""
    df = pd.DataFrame(
        {"a": [1, 2, 3], "b": ["x", "y", "z"], "c": [4.0, 5.0, 6.0]}
    )
    with pytest.raises(ValueError, match="Non-numeric columns found"):
        validate_numeric_columns(df, ["a", "b", "c"])

    # Verify non-numeric column is listed
    try:
        validate_numeric_columns(df, ["a", "b", "c"])
    except ValueError as e:
        error_msg = str(e)
        assert "b" in error_msg


# *********************************************************************************************************************
# Test: validate_ohlc_columns
# *********************************************************************************************************************


def test_validate_ohlc_columns_passes_for_valid_ohlc():
    """Test that validate_ohlc_columns passes for valid OHLC data."""
    df = pd.DataFrame(
        {
            "open": [100.0, 101.0, 102.0],
            "high": [102.0, 103.0, 104.0],
            "low": [99.0, 100.0, 101.0],
            "close": [101.5, 102.5, 103.5],
        }
    )
    # Should not raise
    validate_ohlc_columns(df)


def test_validate_ohlc_columns_raises_for_missing_column():
    """Test that validate_ohlc_columns fails when one OHLC column is missing."""
    df = pd.DataFrame(
        {
            "open": [100.0, 101.0, 102.0],
            "high": [102.0, 103.0, 104.0],
            # Missing 'low' and 'close'
        }
    )
    with pytest.raises(ValueError, match="Missing required columns"):
        validate_ohlc_columns(df)


def test_validate_ohlc_columns_raises_for_non_numeric_ohlc():
    """Test that validate_ohlc_columns raises for non-numeric OHLC columns."""
    df = pd.DataFrame(
        {
            "open": ["a", "b", "c"],
            "high": [102.0, 103.0, 104.0],
            "low": [99.0, 100.0, 101.0],
            "close": [101.5, 102.5, 103.5],
        }
    )
    with pytest.raises(ValueError, match="Non-numeric columns found"):
        validate_ohlc_columns(df)


# *********************************************************************************************************************
# Test: infer_time_column
# *********************************************************************************************************************


def test_infer_time_column_returns_timestamp():
    """Test that infer_time_column returns 'timestamp' when present."""
    df = pd.DataFrame(
        {
            "timestamp": [0, 1, 2],
            "open": [100.0, 101.0, 102.0],
        }
    )
    assert infer_time_column(df) == "timestamp"


def test_infer_time_column_returns_time():
    """Test that infer_time_column returns 'time' when 'timestamp' is absent."""
    df = pd.DataFrame(
        {
            "time": [10, 20, 30],
            "open": [100.0, 101.0, 102.0],
        }
    )
    assert infer_time_column(df) == "time"


def test_infer_time_column_returns_datetime():
    """Test that infer_time_column returns 'datetime' when earlier options absent."""
    df = pd.DataFrame(
        {
            "datetime": ["2020-01-01", "2020-01-02", "2020-01-03"],
            "open": [100.0, 101.0, 102.0],
        }
    )
    assert infer_time_column(df) == "datetime"


def test_infer_time_column_respects_priority():
    """Test that infer_time_column returns first match in priority order."""
    df = pd.DataFrame(
        {
            "timestamp": [0, 1, 2],
            "time": [10, 20, 30],
            "datetime": ["2020-01-01", "2020-01-02", "2020-01-03"],
        }
    )
    # Should return 'timestamp' since it's first in OPTIONAL_TIME_COLUMNS
    assert infer_time_column(df) == "timestamp"


def test_infer_time_column_returns_none_when_absent():
    """Test that infer_time_column returns None when no time column exists."""
    df = pd.DataFrame(
        {
            "open": [100.0, 101.0, 102.0],
            "high": [102.0, 103.0, 104.0],
        }
    )
    assert infer_time_column(df) is None


# *********************************************************************************************************************
# Test: build_feature_contract
# *********************************************************************************************************************


def test_build_feature_contract_default():
    """Test that build_feature_contract returns correct default contract."""
    contract = build_feature_contract()

    assert contract["schema_version"] == "1.0"
    assert contract["required_columns"] == REQUIRED_OHLC_COLUMNS
    assert contract["optional_columns"] == []


def test_build_feature_contract_with_custom_columns():
    """Test build_feature_contract with custom column specifications."""
    custom_required = ["open", "high", "low", "close", "volume"]
    custom_optional = ["timestamp", "rsi", "macd"]

    contract = build_feature_contract(
        required_columns=custom_required,
        optional_columns=custom_optional,
        schema_version="2.0",
    )

    assert contract["schema_version"] == "2.0"
    assert contract["required_columns"] == custom_required
    assert contract["optional_columns"] == custom_optional


def test_build_feature_contract_with_partial_override():
    """Test build_feature_contract with partial override."""
    optional = [OPTIONAL_VOLUME_COLUMNS[0], OPTIONAL_SIGNED_FLOW_COLUMNS[0]]

    contract = build_feature_contract(optional_columns=optional)

    assert contract["schema_version"] == "1.0"
    assert contract["required_columns"] == REQUIRED_OHLC_COLUMNS
    assert contract["optional_columns"] == optional


# *********************************************************************************************************************
# Test: Integration with schema constants
# *********************************************************************************************************************


def test_constants_are_lists():
    """Verify that schema constants are lists."""
    assert isinstance(REQUIRED_OHLC_COLUMNS, list)
    assert isinstance(OPTIONAL_VOLUME_COLUMNS, list)
    assert isinstance(OPTIONAL_SIGNED_FLOW_COLUMNS, list)
    assert isinstance(OPTIONAL_TIME_COLUMNS, list)


def test_required_ohlc_columns_are_expected():
    """Verify REQUIRED_OHLC_COLUMNS contains expected values."""
    assert set(REQUIRED_OHLC_COLUMNS) == {"open", "high", "low", "close"}


def test_optional_time_columns_priority_order():
    """Verify that OPTIONAL_TIME_COLUMNS maintains priority order."""
    assert OPTIONAL_TIME_COLUMNS[0] == "timestamp"
    assert OPTIONAL_TIME_COLUMNS[1] == "time"
    assert OPTIONAL_TIME_COLUMNS[2] == "datetime"
