from __future__                    import annotations
import numpy                       as np
import pandas                      as pd
from signal_analysis.utils.helpers import ema_series, to_series, validate_window

# \file **********************************************************************
# COMPANY:            ELATIN
# PROJECT:            SIGNAL_ANALYSIS
# COMPONENT:          INDICATORS
# MODULE NAME:        momentum.py
# DESCRIPTION:        @brief Momentum indicator utilities
# CREATION DATE:      22.04.2026
# VERSION:            $Revision: 0.2$
# CHANGES:            22.04.2026 - Migrated to short banner comments.
#                     22.04.2026 - Refactored to class-based OOP.
# *****************************************************************************

class MomentumIndicators:
    # ***********************************************************************************************************************
    # Functionname:       MomentumIndicators.compute_rsi(series, window: int = 14, fillna: bool = False)
    #
    # @brief              Compute Relative Strength Index (RSI).
    # @pre                window > 0
    # @post               Returns RSI series named rsi_window.
    # @param[in]          series: Input signal
    #                     window: RSI period
    #                     fillna: Fill NaN with neutral RSI level when True
    # @param[out]         out: RSI as pandas Series
    #
    # @callsequence       @startuml
    #                     title MomentumIndicators.compute_rsi
    #                     start
    #                     :Validate window;
    #                     :Convert input to pandas Series;
    #                     :Compute positive and negative deltas;
    #                     :Apply EMA smoothing to both components;
    #                     :Compute relative strength and RSI;
    #                     if (fillna?) then (yes)
    #                       :Fill NaN with 50.0;
    #                     endif
    #                     :Return RSI series;
    #                     stop
    #                     @enduml
    # ***********************************************************************************************************************
    @staticmethod
    def compute_rsi(series, window: int = 14, fillna: bool = False) -> pd.Series:
        validate_window(window, "window")
        s = to_series(series)
        diff = s.diff(1)
        up_direction = diff.where(diff > 0, 0.0)
        down_direction = -diff.where(diff < 0, 0.0)
        # Use 0 min_periods when fillna so output has no leading NaN.
        min_periods = 0 if fillna else window
        ema_up = up_direction.ewm(alpha=1 / window, min_periods=min_periods, adjust=False).mean()
        ema_down = down_direction.ewm(alpha=1 / window, min_periods=min_periods, adjust=False).mean()
        relative_strength = ema_up / ema_down
        rsi = pd.Series(np.where(ema_down == 0, 100.0, 100.0 - (100.0 / (1.0 + relative_strength))), 
                        index=s.index, name=f"rsi_{window}")
        if fillna:
            rsi = rsi.fillna(50.0)
        return rsi

    # ***********************************************************************************************************************
    # Functionname:       MomentumIndicators.compute_tsi(series, window_slow: int = 25, window_fast: int = 13,
    #                              fillna: bool = False)
    #
    # @brief              Compute True Strength Index (TSI).
    # @pre                window_slow > 0 and window_fast > 0
    # @post               Returns TSI series named tsi_window_slow_window_fast.
    # @param[in]          series: Input signal
    #                     window_slow: Slow smoothing period
    #                     window_fast: Fast smoothing period
    #                     fillna: Fill NaN/inf values when True
    # @param[out]         out: TSI as pandas Series
    #
    # @callsequence       @startuml
    #                     title MomentumIndicators.compute_tsi
    #                     start
    #                     :Validate window_slow and window_fast;
    #                     :Convert input to pandas Series;
    #                     :Compute first derivative;
    #                     :Apply double EMA to momentum;
    #                     :Apply double EMA to absolute momentum;
    #                     :Compute TSI ratio scaled by 100;
    #                     if (fillna?) then (yes)
    #                       :Replace inf/NaN with 0.0;
    #                     endif
    #                     :Return TSI series;
    #                     stop
    #                     @enduml
    # ***********************************************************************************************************************
    @staticmethod
    def compute_tsi(series, window_slow: int = 25, window_fast: int = 13, fillna: bool = False) -> pd.Series:
        validate_window(window_slow, "window_slow")
        validate_window(window_fast, "window_fast")
        s = to_series(series)
        diff = s.diff(1)
        min_periods_slow = 0 if fillna else window_slow
        min_periods_fast = 0 if fillna else window_fast
        # Double-smooth both price momentum and absolute momentum.
        smoothed = ema_series(ema_series(diff, window_slow, adjust=False, min_periods=min_periods_slow),
                              window_fast, adjust=False, min_periods=min_periods_fast)
        smoothed_abs = ema_series(ema_series(diff.abs(), window_slow, adjust=False, min_periods=min_periods_slow),
                                  window_fast, adjust=False, min_periods=min_periods_fast)
        tsi = 100.0 * (smoothed / smoothed_abs)
        tsi = pd.Series(tsi, index=s.index, name=f"tsi_{window_slow}_{window_fast}")
        if fillna:
            tsi = tsi.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return tsi

    # ***********************************************************************************************************************
    # Functionname:       MomentumIndicators.compute_roc(series, window: int = 12, fillna: bool = False)
    #
    # @brief              Compute Rate of Change (ROC) in percent.
    # @pre                window > 0
    # @post               Returns ROC series named roc_window.
    # @param[in]          series: Input signal
    #                     window: Comparison lag
    #                     fillna: Fill NaN/inf values when True
    # @param[out]         out: ROC as pandas Series
    #
    # @callsequence       @startuml
    #                     title MomentumIndicators.compute_roc
    #                     start
    #                     :Validate window;
    #                     :Convert input to pandas Series;
    #                     :Shift series by window periods;
    #                     :Compute percent change * 100;
    #                     if (fillna?) then (yes)
    #                       :Replace inf/NaN with 0.0;
    #                     endif
    #                     :Return ROC series;
    #                     stop
    #                     @enduml
    # ***********************************************************************************************************************
    @staticmethod
    def compute_roc(series, window: int = 12, fillna: bool = False) -> pd.Series:
        validate_window(window, "window")
        s = to_series(series)
        shifted = s.shift(window)
        # Percentage change relative to lagged value.
        roc = ((s - shifted) / shifted) * 100.0
        roc = pd.Series(roc, index=s.index, name=f"roc_{window}")
        if fillna:
            roc = roc.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return roc

    # ***********************************************************************************************************************
    # Functionname:       MomentumIndicators.compute_rsi_tsi_regime(series, rsi_window: int = 14,
    #                              tsi_window_slow: int = 25, tsi_window_fast: int = 13,
    #                              rsi_low: float = 35.0, rsi_high: float = 75.0,
    #                              tsi_long_below: float = 0.0, tsi_short_above: float = 25.0,
    #                              fillna: bool = False)
    #
    # @brief              Compute RSI/TSI and derive simple long/short/flat regime signals.
    # @pre                Valid RSI/TSI windows and rsi_low <= rsi_high.
    # @post               Returns DataFrame with rsi, tsi, rsi_position, and tsi_position.
    # @param[in]          series: Input price series (typically close)
    #                     rsi_window: RSI period
    #                     tsi_window_slow: TSI slow period
    #                     tsi_window_fast: TSI fast period
    #                     rsi_low: Long trigger threshold for RSI
    #                     rsi_high: Short trigger threshold for RSI
    #                     tsi_long_below: Long trigger threshold for TSI
    #                     tsi_short_above: Short trigger threshold for TSI
    #                     fillna: Fill NaN/inf values for RSI/TSI inputs when True
    # @param[out]         out: DataFrame of indicators and discrete positions
    #
    # @callsequence       @startuml
    #                     title MomentumIndicators.compute_rsi_tsi_regime
    #                     start
    #                     if (rsi_low > rsi_high or tsi_long_below > tsi_short_above?) then (yes)
    #                       :Raise ValueError;
    #                       stop
    #                     endif
    #                     :Compute RSI series;
    #                     :Compute TSI series;
    #                     :Apply RSI thresholds to derive rsi_position;
    #                     :Apply TSI thresholds to derive tsi_position;
    #                     :Return combined DataFrame;
    #                     stop
    #                     @enduml
    # ***********************************************************************************************************************
    @staticmethod
    def compute_rsi_tsi_regime(series,
                               rsi_window: int = 14,
                               tsi_window_slow: int = 25,
                               tsi_window_fast: int = 13,
                               rsi_low: float = 35.0,
                               rsi_high: float = 75.0,
                               tsi_long_below: float = 0.0,
                               tsi_short_above: float = 25.0,
                               fillna: bool = False) -> pd.DataFrame:
        if rsi_low > rsi_high:
            raise ValueError("rsi_low must be <= rsi_high")
        if tsi_long_below > tsi_short_above:
            raise ValueError("tsi_long_below must be <= tsi_short_above")

        rsi = MomentumIndicators.compute_rsi(series=series, window=rsi_window, fillna=fillna)
        tsi = MomentumIndicators.compute_tsi(series=series, window_slow=tsi_window_slow, window_fast=tsi_window_fast, fillna=fillna)

        rsi_position = pd.Series(0.0, index=rsi.index, name="rsi_position")
        rsi_position[rsi < rsi_low] = 1.0
        rsi_position[rsi > rsi_high] = -1.0

        tsi_position = pd.Series(0.0, index=tsi.index, name="tsi_position")
        # Mirrors the legacy rules: long when TSI is below lower level, short above upper level.
        tsi_position[tsi < tsi_long_below] = 1.0
        tsi_position[tsi > tsi_short_above] = -1.0

        return pd.DataFrame(
            {
                "rsi": rsi,
                "tsi": tsi,
                "rsi_position": rsi_position,
                "tsi_position": tsi_position,
            },
            index=rsi.index,
        )

# ---------------------------------------------------------------------------
# Backward-compatible module-level wrappers
# ---------------------------------------------------------------------------
def compute_rsi(series, window: int = 14, fillna: bool = False) -> pd.Series:
    return MomentumIndicators.compute_rsi(series, window=window, fillna=fillna)

def compute_tsi(series, window_slow: int = 25, window_fast: int = 13, fillna: bool = False) -> pd.Series:
    return MomentumIndicators.compute_tsi(series, window_slow=window_slow, window_fast=window_fast, fillna=fillna)

def compute_roc(series, window: int = 12, fillna: bool = False) -> pd.Series:
    return MomentumIndicators.compute_roc(series, window=window, fillna=fillna)

def compute_rsi_tsi_regime(
    series,
    rsi_window: int = 14,
    tsi_window_slow: int = 25,
    tsi_window_fast: int = 13,
    rsi_low: float = 35.0,
    rsi_high: float = 75.0,
    tsi_long_below: float = 0.0,
    tsi_short_above: float = 25.0,
    fillna: bool = False,
) -> pd.DataFrame:
    return MomentumIndicators.compute_rsi_tsi_regime(
        series=series,
        rsi_window=rsi_window,
        tsi_window_slow=tsi_window_slow,
        tsi_window_fast=tsi_window_fast,
        rsi_low=rsi_low,
        rsi_high=rsi_high,
        tsi_long_below=tsi_long_below,
        tsi_short_above=tsi_short_above,
        fillna=fillna,
    )

__all__ = ["MomentumIndicators", "compute_rsi", "compute_tsi", "compute_roc", "compute_rsi_tsi_regime"]
