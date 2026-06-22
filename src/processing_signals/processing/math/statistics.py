from __future__ import annotations

import numpy as np
import pandas as pd


def safe_returns(series: pd.Series, method: str = "pct") -> pd.Series:
    series = pd.to_numeric(series, errors="coerce")
    if method == "log":
        return np.log(series / series.shift(1)).replace([np.inf, -np.inf], np.nan)
    return series.pct_change().replace([np.inf, -np.inf], np.nan)


def rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    mean = s.rolling(window).mean()
    std = s.rolling(window).std()
    return (s - mean) / std.replace(0, np.nan)


def rolling_drawdown(series: pd.Series, window: int) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    rolling_max = s.rolling(window).max()
    return (s - rolling_max) / rolling_max.replace(0, np.nan)


def rolling_autocorr(series: pd.Series, window: int, lag: int = 1) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    return s.rolling(window).corr(s.shift(lag))


def rolling_beta(asset: pd.Series, benchmark: pd.Series, window: int) -> pd.Series:
    asset = pd.to_numeric(asset, errors="coerce")
    benchmark = pd.to_numeric(benchmark, errors="coerce")
    cov = asset.rolling(window).cov(benchmark)
    var = benchmark.rolling(window).var()
    return cov / var.replace(0, np.nan)


def rolling_distribution_features(series: pd.Series, windows: list[int]) -> pd.DataFrame:
    s = pd.to_numeric(series, errors="coerce")
    out = pd.DataFrame(index=s.index)

    for window in windows:
        out[f"rolling_mean_{window}"] = s.rolling(window).mean()
        out[f"rolling_std_{window}"] = s.rolling(window).std()
        out[f"rolling_var_{window}"] = s.rolling(window).var()
        out[f"rolling_skewness_{window}"] = s.rolling(window).skew()
        out[f"rolling_kurtosis_{window}"] = s.rolling(window).kurt()
        out[f"rolling_zscore_{window}"] = rolling_zscore(s, window)
        out[f"rolling_min_{window}"] = s.rolling(window).min()
        out[f"rolling_max_{window}"] = s.rolling(window).max()
        out[f"rolling_range_{window}"] = out[f"rolling_max_{window}"] - out[f"rolling_min_{window}"]
        out[f"rolling_q25_{window}"] = s.rolling(window).quantile(0.25)
        out[f"rolling_q75_{window}"] = s.rolling(window).quantile(0.75)
        out[f"rolling_iqr_{window}"] = out[f"rolling_q75_{window}"] - out[f"rolling_q25_{window}"]
        out[f"rolling_autocorr_{window}"] = rolling_autocorr(s, window)
        out[f"rolling_drawdown_{window}"] = rolling_drawdown(s, window)

    return out


def summarize_series(series: pd.Series) -> dict[str, float | None]:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return {
            "mean": None,
            "std": None,
            "var": None,
            "skewness": None,
            "kurtosis": None,
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
        "min": float(s.min()),
        "max": float(s.max()),
        "last": float(s.iloc[-1]),
    }


def last_valid_dict(df: pd.DataFrame) -> dict[str, float | None]:
    result: dict[str, float | None] = {}
    if df.empty:
        return result

    for col in df.columns:
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        result[col] = None if s.empty else float(s.iloc[-1])
    return result
