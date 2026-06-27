from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

DEFAULT_WINDOWS: tuple[int, ...] = (20, 50, 100)
EXCLUDED_COLUMNS: set[str] = {
    "timestamp",
    "timestamp_utc",
    "symbol",
    "timeframe",
    "family_key",
    "data_type",
    "provider",
    "exchange",
    "source_subtype",
    "asset",
    "base_asset",
    "quote_asset",
}


def safe_returns(series: pd.Series, method: str = "pct") -> pd.Series:
    """Return percentage or log returns with infinities converted to NaN."""
    s = pd.to_numeric(series.copy(), errors="coerce")
    if method == "log":
        shifted = s.shift(1)
        valid = (s > 0) & (shifted > 0)
        out = pd.Series(np.nan, index=s.index, dtype="float64")
        out.loc[valid] = np.log(s.loc[valid] / shifted.loc[valid])
        return out.replace([np.inf, -np.inf], np.nan)
    return s.pct_change().replace([np.inf, -np.inf], np.nan)


def detect_numeric_columns(df: pd.DataFrame, excluded_columns: set[str] | None = None) -> list[str]:
    """Detect numeric columns, skipping known metadata columns."""
    excluded = EXCLUDED_COLUMNS if excluded_columns is None else EXCLUDED_COLUMNS.union(excluded_columns)
    columns: list[str] = []

    for column in df.columns:
        if str(column).lower() in excluded:
            continue
        numeric = coerce_numeric_series(df, column)
        if numeric.notna().any():
            columns.append(column)

    return columns


def coerce_numeric_series(df: pd.DataFrame, column: Any) -> pd.Series:
    values = df.loc[:, df.columns == column]
    if isinstance(values, pd.Series):
        raw = values
    elif values.shape[1] == 1:
        raw = values.iloc[:, 0]
    else:
        raw = values.bfill(axis=1).iloc[:, 0]
    return pd.to_numeric(raw, errors="coerce")


def summarize_series(series: pd.Series) -> dict[str, float | None]:
    """Summarize a numeric series with basic moments."""
    s = pd.to_numeric(series.copy(), errors="coerce").dropna()
    if s.empty:
        return {
            "mean": None,
            "std": None,
            "var": None,
            "skewness": None,
            "kurtosis": None,
            "zscore": None,
            "min": None,
            "max": None,
            "last": None,
        }

    return {
        "mean": float(s.mean()),
        "std": float(s.std()) if len(s) > 1 else 0.0,
        "var": float(s.var()) if len(s) > 1 else 0.0,
        "skewness": float(s.skew()) if len(s) > 2 else 0.0,
        "kurtosis": float(s.kurt()) if len(s) > 3 else 0.0,
        "zscore": _last_zscore(s),
        "min": float(s.min()),
        "max": float(s.max()),
        "last": float(s.iloc[-1]),
    }


def last_valid_dict(df: pd.DataFrame) -> dict[str, float | None]:
    """Return last valid numeric value for each DataFrame column."""
    result: dict[str, float | None] = {}
    if df.empty:
        return result

    for column in df.columns:
        s = pd.to_numeric(df[column], errors="coerce").dropna()
        result[str(column)] = None if s.empty else float(s.iloc[-1])

    return result


def _rolling_autocorr(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window, min_periods=window).corr(series.shift(1))


def _rolling_entropy(series: pd.Series, window: int, bins: int = 10) -> pd.Series:
    # Entropy is intentionally omitted from the default per-family payload path:
    # rolling histogram apply is expensive across every block/column. Core
    # contract metrics remain mean/std/var/skewness/kurtosis/zscore/regimes.
    return pd.Series(np.nan, index=series.index, dtype="float64")


def _rolling_drawdown(series: pd.Series, window: int) -> pd.Series:
    rolling_peak = series.rolling(window, min_periods=window).max()
    return (series / rolling_peak) - 1.0


def _rolling_corr(base: pd.Series, ref: pd.Series, window: int) -> pd.Series:
    return base.rolling(window, min_periods=window).corr(ref)


def _rolling_cov(base: pd.Series, ref: pd.Series, window: int) -> pd.Series:
    return base.rolling(window, min_periods=window).cov(ref)


def _rolling_beta(base: pd.Series, ref: pd.Series, window: int) -> pd.Series:
    ref_var = ref.rolling(window, min_periods=window).var().replace(0, np.nan)
    return _rolling_cov(base, ref, window) / ref_var


def _last_zscore(series: pd.Series) -> float:
    if len(series) < 2:
        return 0.0
    std = series.std()
    if pd.isna(std) or std == 0:
        return 0.0
    return float((series.iloc[-1] - series.mean()) / std)


def compute_series_rolling_metrics(
    series: pd.Series,
    windows: list[int],
    reference_series: pd.Series,
) -> pd.DataFrame:
    """Compute pure rolling statistics for a single numeric series."""
    s = pd.to_numeric(series.copy(), errors="coerce")
    ref = pd.to_numeric(reference_series.copy(), errors="coerce")
    returns = safe_returns(s, method="log")
    ref_returns = safe_returns(ref, method="log")

    out = pd.DataFrame(index=s.index)
    for window in windows:
        rolling_obj = s.rolling(window, min_periods=window)
        out[f"rolling_mean_{window}"] = rolling_obj.mean()
        out[f"rolling_std_{window}"] = rolling_obj.std()
        out[f"rolling_var_{window}"] = rolling_obj.var()
        out[f"rolling_skewness_{window}"] = rolling_obj.skew()
        out[f"rolling_kurtosis_{window}"] = rolling_obj.kurt()

        std = out[f"rolling_std_{window}"].replace(0, np.nan)
        out[f"rolling_zscore_{window}"] = (s - out[f"rolling_mean_{window}"]) / std

        out[f"rolling_min_{window}"] = rolling_obj.min()
        out[f"rolling_max_{window}"] = rolling_obj.max()
        out[f"rolling_range_{window}"] = out[f"rolling_max_{window}"] - out[f"rolling_min_{window}"]
        out[f"rolling_quantile_25_{window}"] = rolling_obj.quantile(0.25)
        out[f"rolling_quantile_75_{window}"] = rolling_obj.quantile(0.75)
        out[f"rolling_iqr_{window}"] = (
            out[f"rolling_quantile_75_{window}"] - out[f"rolling_quantile_25_{window}"]
        )

        out[f"rolling_autocorr_{window}"] = _rolling_autocorr(s, window)
        out[f"rolling_entropy_{window}"] = _rolling_entropy(s, window)
        out[f"rolling_drawdown_{window}"] = _rolling_drawdown(s, window)

        ret_std = returns.rolling(window, min_periods=window).std()
        out[f"rolling_volatility_{window}"] = ret_std
        out[f"rolling_realized_volatility_{window}"] = ret_std * np.sqrt(window)

        out[f"rolling_corr_{window}"] = _rolling_corr(returns, ref_returns, window)
        out[f"rolling_cov_{window}"] = _rolling_cov(returns, ref_returns, window)
        out[f"rolling_beta_{window}"] = _rolling_beta(returns, ref_returns, window)

    return out


def compute_block_statistics(
    df: pd.DataFrame,
    windows: list[int] | None = None,
    excluded_columns: set[str] | None = None,
) -> dict[str, Any]:
    """Compute rolling statistics for all numeric columns in a block DataFrame."""
    if df.empty:
        return {
            "numeric_columns": [],
            "reference_column": None,
            "windows": list(DEFAULT_WINDOWS if windows is None else windows),
            "summaries": {},
            "columns": [],
            "last": {},
        }

    active_windows = list(DEFAULT_WINDOWS if windows is None else windows)
    numeric_columns = detect_numeric_columns(df, excluded_columns)

    if not numeric_columns:
        return {
            "numeric_columns": [],
            "reference_column": None,
            "windows": active_windows,
            "summaries": {},
            "columns": [],
            "last": {},
        }

    reference_column = "close" if "close" in numeric_columns else numeric_columns[0]
    reference_series = coerce_numeric_series(df, reference_column)

    feature_frames: list[pd.DataFrame] = []
    summaries: dict[str, Any] = {}

    for column in numeric_columns:
        series = coerce_numeric_series(df, column)
        summaries[column] = summarize_series(series)
        metrics = compute_series_rolling_metrics(series, active_windows, reference_series)
        feature_frames.append(metrics.add_prefix(f"{column}__"))

    feature_frame = pd.concat(feature_frames, axis=1) if feature_frames else pd.DataFrame(index=df.index)

    return {
        "numeric_columns": numeric_columns,
        "reference_column": reference_column,
        "windows": active_windows,
        "summaries": summaries,
        "columns": list(feature_frame.columns),
        "last": last_valid_dict(feature_frame),
    }


__all__ = [
    "DEFAULT_WINDOWS",
    "EXCLUDED_COLUMNS",
    "safe_returns",
    "coerce_numeric_series",
    "detect_numeric_columns",
    "summarize_series",
    "last_valid_dict",
    "compute_series_rolling_metrics",
    "compute_block_statistics",
]
