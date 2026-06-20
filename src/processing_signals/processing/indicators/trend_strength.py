# Compatibility wrapper.
# Canonical implementation lives in signal_analysis.indicators.trend.

from signal_analysis.indicators.trend.trend_strength import (
    TrendStrength,
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

__all__ = [
    "TrendStrength",
    "compute_macd_components",
    "compute_macd",
    "compute_macd_signal",
    "compute_macd_hist",
    "compute_plus_di",
    "compute_minus_di",
    "compute_adx",
    "compute_directional_indicators",
    "compute_trend_helper_signal",
    "classify_trend_block",
    "compute_weighted_trend_score",
    "build_trending_struct",
    "compute_block_trend_profile",
]
