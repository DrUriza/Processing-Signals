from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from processing_signals.processing.math.statistics import (
    DEFAULT_WINDOWS,
    coerce_numeric_series,
    detect_numeric_columns,
    last_valid_dict,
)


def _classify_regime(
    zscore: pd.Series,
    volatility: pd.Series,
    drawdown: pd.Series,
    window: int,
) -> pd.Series:
    high_vol = volatility.rolling(window, min_periods=window).quantile(0.75)
    low_vol = volatility.rolling(window, min_periods=window).quantile(0.25)

    regime = pd.Series("normal", index=zscore.index, dtype="object")
    regime = regime.mask(volatility > high_vol, "high_volatility")
    regime = regime.mask(volatility < low_vol, "low_volatility_compression")
    regime = regime.mask(drawdown <= -0.10, "drawdown_pressure")
    regime = regime.mask(zscore >= 2.0, "positive_outlier")
    regime = regime.mask(zscore <= -2.0, "negative_outlier")
    regime = regime.mask(zscore >= 3.0, "extreme_positive_outlier")
    regime = regime.mask(zscore <= -3.0, "extreme_negative_outlier")
    return regime


def classify_series_regimes(series: pd.Series, windows: list[int]) -> pd.DataFrame:
    """Compute rolling regime labels for one numeric series."""
    s = pd.to_numeric(series.copy(), errors="coerce")
    shifted = s.shift(1)
    valid = (s > 0) & (shifted > 0)
    returns = pd.Series(np.nan, index=s.index, dtype="float64")
    returns.loc[valid] = np.log(s.loc[valid] / shifted.loc[valid])
    returns = returns.replace([np.inf, -np.inf], np.nan)
    out = pd.DataFrame(index=s.index)

    for window in windows:
        mean = s.rolling(window, min_periods=window).mean()
        std = s.rolling(window, min_periods=window).std().replace(0, np.nan)
        zscore = (s - mean) / std

        peak = s.rolling(window, min_periods=window).max()
        drawdown = (s / peak) - 1.0
        volatility = returns.rolling(window, min_periods=window).std()

        out[f"zscore_{window}"] = zscore
        out[f"volatility_{window}"] = volatility
        out[f"drawdown_{window}"] = drawdown
        out[f"regime_{window}"] = _classify_regime(zscore, volatility, drawdown, window)

    return out


def compute_statistical_regimes(
    df: pd.DataFrame,
    windows: list[int] | None = None,
    excluded_columns: set[str] | None = None,
) -> dict[str, Any]:
    """Compute statistical regimes for all detected numeric columns in a block."""
    if df.empty:
        return {
            "numeric_columns": [],
            "windows": list(DEFAULT_WINDOWS if windows is None else windows),
            "columns": [],
            "last": {},
            "last_regimes": {},
        }

    active_windows = list(DEFAULT_WINDOWS if windows is None else windows)
    numeric_columns = detect_numeric_columns(df, excluded_columns)

    if not numeric_columns:
        return {
            "numeric_columns": [],
            "windows": active_windows,
            "columns": [],
            "last": {},
            "last_regimes": {},
        }

    feature_frames: list[pd.DataFrame] = []
    for column in numeric_columns:
        series = coerce_numeric_series(df, column)
        regimes = classify_series_regimes(series, active_windows).add_prefix(f"{column}__")
        feature_frames.append(regimes)

    feature_frame = pd.concat(feature_frames, axis=1) if feature_frames else pd.DataFrame(index=df.index)

    last_regimes: dict[str, str | None] = {}
    for column in feature_frame.columns:
        if "regime_" not in str(column):
            continue
        s = feature_frame[column].dropna()
        last_regimes[str(column)] = None if s.empty else str(s.iloc[-1])

    numeric_only = feature_frame.select_dtypes(include=[np.number])

    return {
        "numeric_columns": numeric_columns,
        "windows": active_windows,
        "columns": list(feature_frame.columns),
        "last": last_valid_dict(numeric_only),
        "last_regimes": last_regimes,
    }


__all__ = [
    "classify_series_regimes",
    "compute_statistical_regimes",
]
