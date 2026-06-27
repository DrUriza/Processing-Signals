from __future__ import annotations

import numpy as np
import pandas as pd

from processing_signals.processing.math.statistics.rolling_distribution import rolling_quantile
from processing_signals.processing.math.statistics.rolling_moments import (
    rolling_kurtosis,
    rolling_mean,
    rolling_skewness,
    rolling_std,
    rolling_zscore,
)
from processing_signals.processing.math.statistics.rolling_risk import rolling_drawdown


def classify_statistical_regime(
    series: pd.Series,
    window: int,
    zscore_threshold: float = 2.0,
    extreme_zscore_threshold: float = 3.0,
    volatility_quantile: float = 0.75,
    compression_quantile: float = 0.25,
    min_periods: int | None = None,
) -> pd.DataFrame:
    """Classify a simple rolling statistical regime."""
    s = pd.to_numeric(series.copy(), errors="coerce")
    required = window if min_periods is None else min_periods

    out = pd.DataFrame(index=s.index)
    out["stat_mean"] = rolling_mean(s, window, min_periods)
    out["stat_std"] = rolling_std(s, window, min_periods)
    out["stat_zscore"] = rolling_zscore(s, window, min_periods)
    out["stat_skewness"] = rolling_skewness(s, window, min_periods)
    out["stat_kurtosis"] = rolling_kurtosis(s, window, min_periods)
    out["stat_drawdown"] = rolling_drawdown(s, window, min_periods)

    high_vol_threshold = rolling_quantile(out["stat_std"], window, volatility_quantile, required)
    compression_threshold = rolling_quantile(out["stat_std"], window, compression_quantile, required)

    regime = pd.Series("normal", index=s.index, dtype="object")
    regime = regime.mask(out["stat_std"] > high_vol_threshold, "high_volatility")
    regime = regime.mask(out["stat_std"] < compression_threshold, "low_volatility_compression")
    regime = regime.mask(out["stat_drawdown"] <= -0.05, "drawdown_pressure")
    regime = regime.mask(out["stat_zscore"] >= zscore_threshold, "positive_outlier")
    regime = regime.mask(out["stat_zscore"] <= -zscore_threshold, "negative_outlier")
    regime = regime.mask(out["stat_zscore"] >= extreme_zscore_threshold, "extreme_positive_outlier")
    regime = regime.mask(out["stat_zscore"] <= -extreme_zscore_threshold, "extreme_negative_outlier")
    out["stat_regime"] = regime
    return out
