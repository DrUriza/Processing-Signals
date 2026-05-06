import pandas                as pd
import pytest
from signal_analysis.filters import (kalman_filter_1d, kalman_filter_multivariate, kalman_filter_ohlc)


def test_kalman_filter_1d_returns_series():
    s = [1, 2, 3, 2, 4, 5, 4]
    out = kalman_filter_1d(s)
    assert isinstance(out, pd.Series)
    assert out.name == "kalman_1d"
    assert len(out) == len(s)


def test_kalman_filter_1d_can_return_velocity():
    s = [1, 2, 3, 2, 4, 5, 4]
    out = kalman_filter_1d(s, return_velocity=True)
    assert isinstance(out, pd.DataFrame)
    assert list(out.columns) == ["estimate", "velocity"]


def test_kalman_filter_multivariate_returns_dataframe():
    df = pd.DataFrame(
        {
            "a": [1, 2, 3, 4, 5],
            "b": [2, 3, 4, 5, 6],
        }
    )
    out = kalman_filter_multivariate(df)
    assert isinstance(out, pd.DataFrame)
    assert list(out.columns) == ["a", "b"]


def test_kalman_filter_ohlc_returns_dataframe():
    df = pd.DataFrame(
        {
            "open": [1, 2, 3, 4, 5],
            "high": [2, 3, 4, 5, 6],
            "low": [0, 1, 2, 3, 4],
            "close": [1.5, 2.5, 3.5, 4.5, 5.5],
        }
    )
    out = kalman_filter_ohlc(df)
    assert isinstance(out, pd.DataFrame)
    assert list(out.columns) == ["open", "high", "low", "close"]


def test_kalman_filter_1d_empty_input_raises_value_error():
    with pytest.raises(ValueError, match="must not be empty"):
        kalman_filter_1d([])


def test_kalman_filter_multivariate_empty_input_raises_value_error():
    df = pd.DataFrame({"a": [], "b": []})
    with pytest.raises(ValueError, match="must not be empty"):
        kalman_filter_multivariate(df)
