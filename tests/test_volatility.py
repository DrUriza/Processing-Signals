import pandas as pd

from signal_analysis.indicators import (
    compute_adx_directional_signal,
    compute_atr,
    compute_bollinger_bands,
    compute_bollinger_reference_signal,
    compute_tr,
)


def test_compute_bollinger_bands_returns_dataframe():
    s = pd.Series([1, 2, 3, 4, 5, 6, 7, 8])
    out = compute_bollinger_bands(s, window=3, n_std=2.0)
    assert isinstance(out, pd.DataFrame)
    assert list(out.columns) == [
        "bb_middle",
        "bb_upper",
        "bb_lower",
        "bb_bandwidth",
        "bb_percent_b",
    ]


def test_compute_tr_returns_series():
    high = pd.Series([2, 3, 4, 5])
    low = pd.Series([1, 2, 3, 4])
    close = pd.Series([1.5, 2.5, 3.5, 4.5])
    out = compute_tr(high, low, close)
    assert isinstance(out, pd.Series)
    assert out.name == "true_range"
    assert len(out) == len(close)


def test_compute_atr_returns_series():
    high = pd.Series([2, 3, 4, 5, 6, 7])
    low = pd.Series([1, 2, 3, 4, 5, 6])
    close = pd.Series([1.5, 2.5, 3.5, 4.5, 5.5, 6.5])
    out = compute_atr(high, low, close, window=3)
    assert isinstance(out, pd.Series)
    assert out.name == "atr_3"
    assert len(out) == len(close)


def test_compute_adx_directional_signal_returns_dataframe():
    high = pd.Series([2, 3, 4, 5, 6, 7, 8, 9])
    low = pd.Series([1, 2, 3, 4, 5, 6, 7, 8])
    close = pd.Series([1.5, 2.4, 3.3, 4.1, 5.0, 6.2, 7.1, 8.0])

    out = compute_adx_directional_signal(high, low, close, window=3)
    assert isinstance(out, pd.DataFrame)
    assert list(out.columns) == ["adx", "md_plus", "md_minus", "md_position"]
    assert len(out) == len(close)


def test_compute_bollinger_reference_signal_returns_dataframe():
    close = pd.Series([10, 11, 12, 11, 13, 14, 13, 15])
    y_pred = pd.Series([10.5, 10.7, 11.2, 11.4, 12.1, 12.8, 13.2, 13.7])

    out = compute_bollinger_reference_signal(close, y_pred, window=3)
    assert isinstance(out, pd.DataFrame)
    assert list(out.columns) == [
        "bb_middle",
        "bb_upper",
        "bb_lower",
        "bb_bandwidth",
        "bb_percent_b",
        "bb_signal",
        "bb_position",
    ]
    assert len(out) == len(close)
