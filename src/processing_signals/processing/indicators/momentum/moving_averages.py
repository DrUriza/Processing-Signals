from __future__ import annotations
import numpy    as np
import pandas   as pd
from signal_analysis.utils.helpers import ema_series, to_series, validate_window

# \file **********************************************************************
# COMPANY:            ELATIN
# PROJECT:            SIGNAL_ANALYSIS
# COMPONENT:          INDICATORS
# MODULE NAME:        moving_averages.py
# DESCRIPTION:        @brief Moving average indicator utilities
# CREATION DATE:      22.04.2026
# VERSION:            $Revision: 0.2$
# CHANGES:            22.04.2026 - Migrated to short banner comments.
#                     22.04.2026 - Refactored to class-based OOP.
# *****************************************************************************

class MovingAverages:
    # ***********************************************************************************************************************
    # Functionname:       MovingAverages.compute_sma(series, window: int = 14, min_periods: int | None = None)
    #
    # @brief              Compute Simple Moving Average (SMA).
    # @pre                window > 0
    # @post               Returns SMA series named sma_window.
    # @param[in]          series: Input signal
    #                     window: SMA period
    #                     min_periods: Minimum periods for rolling mean
    # @param[out]         out: SMA as pandas Series
    #
    # @callsequence       @startuml
    #                     title MovingAverages.compute_sma
    #                     start
    #                     :Validate window;
    #                     :Convert input to pandas Series;
    #                     :Resolve rolling min_periods;
    #                     :Compute rolling mean;
    #                     :Set output series name;
    #                     :Return SMA series;
    #                     stop
    #                     @enduml
    # ***********************************************************************************************************************
    @staticmethod
    def compute_sma(series, window: int = 14, min_periods: int | None = None) -> pd.Series:
        validate_window(window, "window")
        s = to_series(series)
        mp = window if min_periods is None else min_periods
        # Plain arithmetic mean over rolling window.
        sma = s.rolling(window=window, min_periods=mp).mean()
        sma.name = f"sma_{window}"
        return sma

    # ***********************************************************************************************************************
    # Functionname:       MovingAverages.compute_ema(series, window: int = 14, adjust: bool = False,
    #                              min_periods: int | None = None)
    #
    # @brief              Compute Exponential Moving Average (EMA).
    # @pre                window > 0
    # @post               Returns EMA series named ema_window.
    # @param[in]          series: Input signal
    #                     window: EMA period
    #                     adjust: Pandas EWM adjust option
    #                     min_periods: Minimum periods for EWM
    # @param[out]         out: EMA as pandas Series
    #
    # @callsequence       @startuml
    #                     title MovingAverages.compute_ema
    #                     start
    #                     :Convert input to pandas Series;
    #                     :Compute EMA with helper ema_series;
    #                     :Set output series name;
    #                     :Return EMA series;
    #                     stop
    #                     @enduml
    # ***********************************************************************************************************************
    @staticmethod
    def compute_ema(series, window: int = 14, adjust: bool = False, min_periods: int | None = None) -> pd.Series:
        s = to_series(series)
        ema = ema_series(s, window=window, adjust=adjust, min_periods=min_periods)
        ema.name = f"ema_{window}"
        return ema

    # ***********************************************************************************************************************
    # Functionname:       MovingAverages.compute_wma(series, window: int = 9, min_periods: int | None = None)
    #
    # @brief              Compute linearly weighted moving average (WMA).
    # @pre                window > 0
    # @post               Returns WMA series named wma_window.
    # @param[in]          series: Input signal
    #                     window: WMA period
    #                     min_periods: Minimum periods for rolling apply
    # @param[out]         out: WMA as pandas Series
    #
    # @callsequence       @startuml
    #                     title MovingAverages.compute_wma
    #                     start
    #                     :Validate window;
    #                     :Convert input to pandas Series;
    #                     :Build normalized linear weights;
    #                     :Apply weighted rolling function;
    #                     :Set output series name;
    #                     :Return WMA series;
    #                     stop
    #                     @enduml
    # ***********************************************************************************************************************
    @staticmethod
    def compute_wma(series, window: int = 9, min_periods: int | None = None) -> pd.Series:
        validate_window(window, "window")
        s = to_series(series)
        mp = window if min_periods is None else min_periods

        # Precompute normalized linear weights [1, 2, ..., window].
        weights = np.arange(1, window + 1, dtype=float)
        weights = weights / weights.sum()

        def _weighted_avg(values: np.ndarray) -> float:
            return float(np.dot(values, weights))

        wma = s.rolling(window=window, min_periods=mp).apply(_weighted_avg, raw=True)
        wma.name = f"wma_{window}"
        return wma

    # ***********************************************************************************************************************
    # Functionname:       MovingAverages.compute_kama(series, er_window: int = 10, fast: int = 2, slow: int = 30)
    #
    # @brief              Compute Kaufman's Adaptive Moving Average (KAMA).
    # @pre                er_window > 0, fast > 0, slow > 0
    # @post               Returns KAMA series named kama_er_window_fast_slow.
    # @param[in]          series: Input signal
    #                     er_window: Efficiency ratio window
    #                     fast: Fastest EMA equivalent period
    #                     slow: Slowest EMA equivalent period
    # @param[out]         out: KAMA as pandas Series
    #
    # @callsequence       @startuml
    #                     title MovingAverages.compute_kama
    #                     start
    #                     :Validate er_window, fast, and slow;
    #                     :Convert input to pandas Series;
    #                     :Compute efficiency ratio;
    #                     :Build adaptive smoothing constants;
    #                     :Initialize recursive KAMA array;
    #                     :Iterate and update KAMA values;
    #                     :Return KAMA series;
    #                     stop
    #                     @enduml
    # ***********************************************************************************************************************
    @staticmethod
    def compute_kama(series, er_window: int = 10, fast: int = 2, slow: int = 30) -> pd.Series:
        validate_window(er_window, "er_window")
        validate_window(fast, "fast")
        validate_window(slow, "slow")
        s = to_series(series)
        values = s.to_numpy(dtype=float)
        change = np.abs(s - s.shift(er_window))
        volatility = s.diff().abs().rolling(er_window, min_periods=er_window).sum()
        efficiency_ratio = change / volatility
        efficiency_ratio = efficiency_ratio.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        # Smoothing constants derived from fast and slow EMA periods.
        fast_sc = 2.0 / (fast + 1.0)
        slow_sc = 2.0 / (slow + 1.0)
        smoothing_constant = (efficiency_ratio * (fast_sc - slow_sc) + slow_sc) ** 2
        kama = np.full(len(values), np.nan, dtype=float)
        first_valid_idx = s.first_valid_index()
        if first_valid_idx is None:
            return pd.Series(kama, index=s.index, name=f"kama_{er_window}_{fast}_{slow}")
        first_pos = s.index.get_loc(first_valid_idx)
        kama[first_pos] = values[first_pos]
        # Recursive adaptive update with dynamic smoothing coefficient.
        for i in range(first_pos + 1, len(values)):
            if np.isnan(values[i]):
                kama[i] = np.nan
                continue
            prev_kama = kama[i - 1]
            if np.isnan(prev_kama):
                prev_kama = values[i - 1]
            sc = float(smoothing_constant.iloc[i]) if not np.isnan(smoothing_constant.iloc[i]) else 0.0
            kama[i] = prev_kama + sc * (values[i] - prev_kama)
        kama_series = pd.Series(kama, index=s.index, name=f"kama_{er_window}_{fast}_{slow}")
        return kama_series

# ---------------------------------------------------------------------------
# Backward-compatible module-level wrappers
# ---------------------------------------------------------------------------
def compute_sma(series, window: int = 14, min_periods: int | None = None) -> pd.Series:
    return MovingAverages.compute_sma(series, window=window, min_periods=min_periods)

def compute_ema(series, window: int = 14, adjust: bool = False, min_periods: int | None = None) -> pd.Series:
    return MovingAverages.compute_ema(series, window=window, adjust=adjust, min_periods=min_periods)

def compute_wma(series, window: int = 9, min_periods: int | None = None) -> pd.Series:
    return MovingAverages.compute_wma(series, window=window, min_periods=min_periods)

def compute_kama(series, er_window: int = 10, fast: int = 2, slow: int = 30) -> pd.Series:
    return MovingAverages.compute_kama(series, er_window=er_window, fast=fast, slow=slow)

__all__ = ["MovingAverages", "compute_sma", "compute_ema", "compute_wma", "compute_kama"]
