import pandas as pd

from signal_analysis.indicators import (
    VolumeIndicators,
    add_volume_features,
    compute_volume_stats,
    compute_vwap_metrics,
    compute_order_flow_metrics,
)


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open": [10, 11, 12, 13, 12, 13],
            "high": [11, 12, 13, 14, 13, 14],
            "low": [9, 10, 11, 12, 11, 12],
            "close": [10.5, 11.5, 12.5, 12.8, 12.2, 13.5],
            "volume": [100, 120, 140, 160, 180, 220],
            "buy_volume": [55, 70, 78, 90, 92, 120],
            "sell_volume": [45, 50, 62, 70, 88, 100],
        }
    )


def test_compute_volume_stats_returns_expected_columns():
    df = _sample_df()
    out = compute_volume_stats(df, window=3)

    assert isinstance(out, pd.DataFrame)
    assert {"vol_sma_3", "vol_ema_3", "rvol_3", "vol_zscore_3", "obv"}.issubset(set(out.columns))
    assert len(out) == len(df)


def test_rvol_series_name_and_type():
    df = _sample_df()
    out = compute_volume_stats(df, window=3)
    assert isinstance(out["rvol_3"], pd.Series)
    assert len(out["rvol_3"]) == len(df)


def test_compute_vwap_metrics_returns_expected_columns():
    df = _sample_df()
    out = compute_vwap_metrics(df, rolling_window=3, mfi_window=3, cmf_window=3)

    assert isinstance(out, pd.DataFrame)
    assert {"typical_price", "vwap", "rolling_vwap_3", "close_vwap_dist", "mfi_3", "cmf_3"}.issubset(set(out.columns))
    assert len(out) == len(df)


def test_compute_order_flow_metrics_with_flow_cols():
    df = _sample_df()
    out = compute_order_flow_metrics(df)

    assert isinstance(out, pd.DataFrame)
    assert {"volume_delta", "volume_imbalance", "cvd"}.issubset(set(out.columns))
    assert len(out) == len(df)


def test_compute_vwap_series_is_consistent():
    df = _sample_df()
    out_fn  = compute_vwap_metrics(df)["vwap"]
    out_cls = VolumeIndicators.compute_vwap_metrics(df=df)["vwap"]

    assert isinstance(out_fn, pd.Series)
    assert out_fn.equals(out_cls)


def test_add_volume_features_adds_expected_columns():
    df = _sample_df()
    out = add_volume_features(df, vol_window=3, rolling_window=3, mfi_window=3, cmf_window=3, flow_enabled=True)

    expected = {
        "vol_sma_3",
        "vol_ema_3",
        "rvol_3",
        "vol_zscore_3",
        "obv",
        "typical_price",
        "vwap",
        "rolling_vwap_3",
        "close_vwap_dist",
        "mfi_3",
        "cmf_3",
        "volume_delta",
        "volume_imbalance",
        "cvd",
        "breakout_volume_score",
    }

    assert expected.issubset(set(out.columns))
