from __future__ import annotations

import numpy as np
import pandas as pd

from processing_signals.processing.math.statistics.rolling_beta import rolling_beta
from processing_signals.processing.math.statistics.rolling_correlation import rolling_autocorr, rolling_correlation
from processing_signals.processing.math.statistics.rolling_distribution import (
    rolling_entropy,
    rolling_iqr,
    rolling_max,
    rolling_min,
    rolling_quantile,
    rolling_range,
)
from processing_signals.processing.math.statistics.rolling_moments import (
    rolling_kurtosis,
    rolling_mean,
    rolling_skewness,
    rolling_std,
    rolling_var,
    rolling_zscore,
)
from processing_signals.processing.math.statistics.rolling_risk import rolling_drawdown, rolling_max_drawdown
from processing_signals.processing.math.statistics.statistical_regimes import classify_statistical_regime


def safe_returns(series: pd.Series, method: str = "pct") -> pd.Series:
    """Return percentage or log returns with infinities converted to NaN."""
    s = pd.to_numeric(series.copy(), errors="coerce")
    if method == "log":
        return np.log(s / s.shift(1)).replace([np.inf, -np.inf], np.nan)
    return s.pct_change().replace([np.inf, -np.inf], np.nan)


def rolling_distribution_features(series: pd.Series, windows: list[int]) -> pd.DataFrame:
    """Build the legacy rolling distribution feature frame used by ProcessingMathEngine."""
    s = pd.to_numeric(series.copy(), errors="coerce")
    out = pd.DataFrame(index=s.index)

    for window in windows:
        out[f"rolling_mean_{window}"] = rolling_mean(s, window)
        out[f"rolling_std_{window}"] = rolling_std(s, window)
        out[f"rolling_var_{window}"] = rolling_var(s, window)
        out[f"rolling_skewness_{window}"] = rolling_skewness(s, window)
        out[f"rolling_kurtosis_{window}"] = rolling_kurtosis(s, window)
        out[f"rolling_zscore_{window}"] = rolling_zscore(s, window)
        out[f"rolling_min_{window}"] = rolling_min(s, window)
        out[f"rolling_max_{window}"] = rolling_max(s, window)
        out[f"rolling_range_{window}"] = rolling_range(s, window)
        out[f"rolling_q25_{window}"] = rolling_quantile(s, window, 0.25)
        out[f"rolling_q75_{window}"] = rolling_quantile(s, window, 0.75)
        out[f"rolling_iqr_{window}"] = rolling_iqr(s, window)
        out[f"rolling_autocorr_{window}"] = rolling_autocorr(s, window)
        out[f"rolling_drawdown_{window}"] = rolling_drawdown(s, window)

    return out


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
    """Return the last valid numeric value for each DataFrame column."""
    result: dict[str, float | None] = {}
    if df.empty:
        return result

    for col in df.columns:
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        result[col] = None if s.empty else float(s.iloc[-1])
    return result


__all__ = [
    "rolling_mean",
    "rolling_std",
    "rolling_var",
    "rolling_skewness",
    "rolling_kurtosis",
    "rolling_zscore",
    "rolling_quantile",
    "rolling_min",
    "rolling_max",
    "rolling_range",
    "rolling_iqr",
    "rolling_entropy",
    "rolling_autocorr",
    "rolling_correlation",
    "rolling_drawdown",
    "rolling_max_drawdown",
    "rolling_beta",
    "classify_statistical_regime",
    "safe_returns",
    "rolling_distribution_features",
    "summarize_series",
    "last_valid_dict",
]
