# Compatibility wrapper.
# Canonical implementation lives in signal_analysis.indicators.volatility.

from signal_analysis.indicators.volatility.volatility import (
    VolatilityIndicators,
    compute_adx_directional_signal,
    compute_atr,
    compute_bollinger_bands,
    compute_bollinger_reference_signal,
    compute_tr,
)

__all__ = [
    "VolatilityIndicators",
    "compute_bollinger_bands",
    "compute_tr",
    "compute_atr",
    "compute_adx_directional_signal",
    "compute_bollinger_reference_signal",
]