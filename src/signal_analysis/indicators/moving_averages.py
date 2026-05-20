# Compatibility wrapper.
# Canonical implementation lives in signal_analysis.indicators.momentum.

from signal_analysis.indicators.momentum.moving_averages import (
    MovingAverages,
    compute_ema,
    compute_kama,
    compute_sma,
    compute_wma,
)

__all__ = ["MovingAverages", "compute_sma", "compute_ema", "compute_wma", "compute_kama"]
