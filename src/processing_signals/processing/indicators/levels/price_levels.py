from __future__   import annotations
import numpy      as np
import pandas     as pd
from scipy.signal import argrelextrema

from signal_analysis.utils.helpers import to_series

# \file **********************************************************************
# COMPANY:            ELATIN
# PROJECT:            SIGNAL_ANALYSIS
# COMPONENT:          INDICATORS
# MODULE NAME:        price_levels.py
# DESCRIPTION:        @brief Fibonacci and support/resistance level indicators
# CREATION DATE:      22.04.2026
# VERSION:            $Revision: 0.1$
# CHANGES:            22.04.2026 - Added Fibonacci and support/resistance indicators.
# *****************************************************************************

class PriceLevelIndicators:
    # ***********************************************************************************************************************
    # Functionname:       PriceLevelIndicators.compute_fibonacci_levels(high, low)
    #
    # @brief              Compute canonical Fibonacci retracement levels.
    # @pre                high and low are aligned series-like inputs.
    # @post               Returns 0.0..1.0 retracement dictionary as a Series.
    # @param[in]          high: High values
    #                     low: Low values
    # @param[out]         out: Fibonacci levels series
    #
    # @callsequence       @startuml
    #                     title PriceLevelIndicators.compute_fibonacci_levels
    #                     start
    #                     :Convert high and low to pandas Series;
    #                     :Compute swing_high and swing_low;
    #                     :Compute retracement span;
    #                     :Build Fibonacci levels dictionary;
    #                     :Return Fibonacci levels series;
    #                     stop
    #                     @enduml
    # ***********************************************************************************************************************
    @staticmethod
    def compute_fibonacci_levels(high,low) -> pd.Series:
        high_series = to_series(high, name="high")
        low_series  = to_series(low, name="low")

        swing_high = float(high_series.max())
        swing_low  = float(low_series.min())
        span       = swing_high - swing_low

        levels = {"0.0": swing_low,
                  "0.236": swing_low + span * 0.236,
                  "0.382": swing_low + span * 0.382,
                  "0.5": swing_low + span * 0.5,
                  "0.618": swing_low + span * 0.618,
                  "0.786": swing_low + span * 0.786,
                  "1.0": swing_high}
        return pd.Series(levels, name="fibonacci_levels", dtype=float)

    # ***********************************************************************************************************************
    # Functionname:       PriceLevelIndicators.compute_support_resistance(series, max_levels: int = 5,
    #                              sensitivity: int = 3)
    #
    # @brief              Compute support and resistance levels from local extrema.
    # @pre                series is numeric and max_levels/sensitivity > 0.
    # @post               Returns dict with support and resistance float arrays.
    # @param[in]          series: Input price series, usually close
    #                     max_levels: Max number of levels returned per side
    #                     sensitivity: Argrelextrema order parameter
    # @param[out]         out: Support/resistance dictionary
    #
    # @callsequence       @startuml
    #                     title PriceLevelIndicators.compute_support_resistance
    #                     start
    #                     if (max_levels <= 0 or not int?) then (yes)
    #                       :Raise ValueError;
    #                       stop
    #                     endif
    #                     if (sensitivity <= 0 or not int?) then (yes)
    #                       :Raise ValueError;
    #                       stop
    #                     endif
    #                     :Convert series to pandas Series;
    #                     :Find local maxima and minima with argrelextrema;
    #                     :Build resistance and support lists;
    #                     :Return support/resistance dictionary;
    #                     stop
    #                     @enduml
    # ***********************************************************************************************************************
    @staticmethod
    def compute_support_resistance(series,
                                   max_levels: int = 5,
                                   sensitivity: int = 3) -> dict[str, list[float]]:
        if not isinstance(max_levels, int) or max_levels <= 0:
            raise ValueError("max_levels must be a positive integer.")
        if not isinstance(sensitivity, int) or sensitivity <= 0:
            raise ValueError("sensitivity must be a positive integer.")

        s = to_series(series, name="price")
        values = s.to_numpy(dtype=float)

        pivots_high = argrelextrema(values, np.greater, order=sensitivity)[0]
        pivots_low = argrelextrema(values, np.less, order=sensitivity)[0]

        resistance = sorted([float(values[i]) for i in pivots_high])[-max_levels:]
        support = sorted([float(values[i]) for i in pivots_low])[:max_levels]

        return {"support": support, "resistance": resistance}

# ---------------------------------------------------------------------------
# Backward-compatible module-level wrappers
# ---------------------------------------------------------------------------
def compute_fibonacci_levels(high,low) -> pd.Series:
    return PriceLevelIndicators.compute_fibonacci_levels(high, low)

def compute_support_resistance(series,max_levels: int = 5,sensitivity: int = 3) -> dict[str, list[float]]:
    return PriceLevelIndicators.compute_support_resistance(series,
                                                           max_levels=max_levels,
                                                           sensitivity=sensitivity)

__all__ = ["PriceLevelIndicators", "compute_fibonacci_levels", "compute_support_resistance"]
