import pandas as pd

from signal_analysis.indicators import (
    build_trending_struct,
    classify_trend_block,
    compute_adx,
    compute_block_trend_profile,
    compute_directional_indicators,
    compute_macd,
    compute_macd_components,
    compute_macd_hist,
    compute_macd_signal,
    compute_minus_di,
    compute_plus_di,
    compute_trend_helper_signal,
    compute_weighted_trend_score,
)


def test_compute_macd_components_returns_dataframe():
    s = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9])
    out = compute_macd_components(s, window_slow=6, window_fast=3, window_signal=2)
    assert isinstance(out, pd.DataFrame)
    assert list(out.columns) == ["macd", "macd_signal", "macd_hist"]


def test_compute_macd_series_exports():
    s = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9])
    macd = compute_macd(s, window_slow=6, window_fast=3, window_signal=2)
    sig = compute_macd_signal(s, window_slow=6, window_fast=3, window_signal=2)
    hist = compute_macd_hist(s, window_slow=6, window_fast=3, window_signal=2)
    assert isinstance(macd, pd.Series)
    assert isinstance(sig, pd.Series)
    assert isinstance(hist, pd.Series)


def test_compute_directional_indicators_returns_dataframe():
    high = pd.Series([2, 3, 4, 5, 6, 7, 8])
    low = pd.Series([1, 2, 3, 4, 5, 6, 7])
    close = pd.Series([1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5])

    out = compute_directional_indicators(high, low, close, window=3)
    assert isinstance(out, pd.DataFrame)
    assert list(out.columns) == ["plus_di", "minus_di", "dx", "adx"]


def test_adx_and_di_wrappers_return_series():
    high = pd.Series([2, 3, 4, 5, 6, 7, 8])
    low = pd.Series([1, 2, 3, 4, 5, 6, 7])
    close = pd.Series([1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5])

    plus_di = compute_plus_di(high, low, close, window=3)
    minus_di = compute_minus_di(high, low, close, window=3)
    adx = compute_adx(high, low, close, window=3)

    assert isinstance(plus_di, pd.Series)
    assert isinstance(minus_di, pd.Series)
    assert isinstance(adx, pd.Series)


def test_compute_trend_helper_signal_returns_dataframe():
    high = pd.Series([2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    low = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    close = pd.Series([1.5, 2.5, 3.5, 4.5, 5.2, 6.1, 6.8, 7.9, 9.1, 10.2])

    out = compute_trend_helper_signal(
        high,
        low,
        close,
        adx_window=3,
        adx_threshold=10.0,
        macd_window_slow=6,
        macd_window_fast=3,
        macd_window_signal=2,
    )
    assert isinstance(out, pd.DataFrame)
    assert list(out.columns) == ["macd", "macd_signal", "macd_hist", "plus_di", "minus_di", "adx", "position"]
    assert len(out) == len(close)
    assert set(out["position"].dropna().unique()).issubset({-1.0, 0.0, 1.0})


def test_compute_trend_helper_signal_negative_adx_threshold_raises_value_error():
    high = pd.Series([2, 3, 4, 5])
    low = pd.Series([1, 2, 3, 4])
    close = pd.Series([1.5, 2.5, 3.5, 4.5])

    try:
        compute_trend_helper_signal(high, low, close, adx_threshold=-1.0)
        assert False, "Expected ValueError for negative adx_threshold"
    except ValueError as exc:
        assert "adx_threshold" in str(exc)


def test_classify_trend_block_returns_expected_label():
    block = pd.DataFrame({"open": [10.0, 10.2], "close": [10.1, 11.0]})
    assert classify_trend_block(block) == "alcista"


def test_compute_weighted_trend_score_updates_history_and_returns_float():
    history = ["lateral", "alcista"]
    score = compute_weighted_trend_score("bajista", history)
    assert isinstance(score, float)
    assert history[-1] == "bajista"
    assert -1.0 <= score <= 1.0


def test_build_trending_struct_returns_levels_and_signal():
    df = pd.DataFrame({"FullTrend": [-0.7, -0.2, 0.1, 0.5, 0.9]})
    out = build_trending_struct(df)
    assert isinstance(out, dict)
    assert set(out.keys()) == {"levels", "last_value", "signal"}
    assert set(out["levels"].keys()) == {"strong_buy", "buy", "sell", "strong_sell"}


def test_compute_block_trend_profile_returns_series_with_same_length():
    df = pd.DataFrame(
        {
            "open": [10, 10.1, 10.2, 10.3, 10.2, 10.4],
            "close": [10.1, 10.2, 10.15, 10.5, 10.6, 10.7],
        }
    )
    out = compute_block_trend_profile(df, step=2, sensitivity=10.0)
    assert isinstance(out, pd.Series)
    assert len(out) == len(df)
    assert out.name == "trend_profile_2"
