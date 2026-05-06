from __future__                          import annotations
import numpy                             as np
import pandas                            as pd
from signal_analysis.indicators.momentum import compute_rsi
from signal_analysis.utils.helpers       import to_series, validate_window


# \file **********************************************************************
# COMPANY:            ELATIN
# PROJECT:            SIGNAL_ANALYSIS
# COMPONENT:          INDICATORS
# MODULE NAME:        oscillators.py
# DESCRIPTION:        @brief Oscillator indicator utilities
# CREATION DATE:      22.04.2026
# VERSION:            $Revision: 0.2$
# CHANGES:            22.04.2026 - Migrated to short banner comments.
#                     22.04.2026 - Refactored to class-based OOP.
# *****************************************************************************

class Oscillators:
    # ***********************************************************************************************************************
    # Functionname:       Oscillators.compute_stochastic(high, low, close, window: int = 14,
    #                              smooth_window: int = 3, min_periods: int | None = None)
    #
    # @brief              Compute Stochastic Oscillator values %K and %D.
    # @pre                window > 0 and smooth_window > 0
    # @post               Returns DataFrame with stoch_k and stoch_d columns.
    # @param[in]          high: High price series
    #                     low: Low price series
    #                     close: Close price series
    #                     window: Rolling range window
    #                     smooth_window: Smoothing window for %D
    #                     min_periods: Minimum periods for rolling min/max
    # @param[out]         out: Stochastic DataFrame
    #
    # @callsequence       @startuml
    #                     title Oscillators.compute_stochastic
    #                     start
    #                     :Validate window and smooth_window;
    #                     :Convert high, low, and close to pandas Series;
    #                     :Resolve rolling min_periods;
    #                     :Compute rolling low and high bounds;
    #                     :Compute stoch_k and smoothed stoch_d;
    #                     :Return stochastic DataFrame;
    #                     stop
    #                     @enduml
    # ***********************************************************************************************************************
    @staticmethod
    def compute_stochastic(high, low, close, window: int = 14, smooth_window: int = 3, min_periods: int | None = None) -> pd.DataFrame:
        validate_window(window, "window")
        validate_window(smooth_window, "smooth_window")

        h = to_series(high, name="high")
        l = to_series(low, name="low")
        c = to_series(close, name="close")

        mp = window if min_periods is None else min_periods

        # Rolling high/low range for %K normalization.
        low_min = l.rolling(window=window, min_periods=mp).min()
        high_max = h.rolling(window=window, min_periods=mp).max()

        denom = (high_max - low_min).replace(0.0, np.nan)
        stoch_k = 100.0 * (c - low_min) / denom
        stoch_d = stoch_k.rolling(window=smooth_window, min_periods=smooth_window).mean()

        return pd.DataFrame(
            {
                "stoch_k": stoch_k,
                "stoch_d": stoch_d,
            },
            index=c.index,
        )

    # ***********************************************************************************************************************
    # Functionname:       Oscillators.compute_stochastic_signal(high, low, close, window: int = 14,
    #                              smooth_window: int = 3, min_periods: int | None = None)
    #
    # @brief              Return only Stochastic %D signal line.
    # @pre                Same preconditions as compute_stochastic.
    # @post               Returns stoch_d series.
    # @param[in]          high: High price series
    #                     low: Low price series
    #                     close: Close price series
    #                     window: Rolling range window
    #                     smooth_window: Smoothing window for %D
    #                     min_periods: Minimum periods for rolling min/max
    # @param[out]         out: Stochastic %D series
    #
    # @callsequence       @startuml
    #                     title Oscillators.compute_stochastic_signal
    #                     start
    #                     :Call compute_stochastic;
    #                     :Extract stoch_d column;
    #                     :Return signal series;
    #                     stop
    #                     @enduml
    # ***********************************************************************************************************************
    @staticmethod
    def compute_stochastic_signal(high, low, close, window: int = 14, smooth_window: int = 3, min_periods: int | None = None) -> pd.Series:
        return Oscillators.compute_stochastic(high=high, low=low, close=close, window=window, 
                                              smooth_window=smooth_window, min_periods=min_periods)["stoch_d"]

    # ***********************************************************************************************************************
    # Functionname:       Oscillators.compute_williams_r(high, low, close, window: int = 14,
    #                              min_periods: int | None = None)
    #
    # @brief              Compute Williams %R oscillator.
    # @pre                window > 0
    # @post               Returns series named williams_r_window.
    # @param[in]          high: High price series
    #                     low: Low price series
    #                     close: Close price series
    #                     window: Rolling range window
    #                     min_periods: Minimum periods for rolling min/max
    # @param[out]         out: Williams %R series
    #
    # @callsequence       @startuml
    #                     title Oscillators.compute_williams_r
    #                     start
    #                     :Validate window;
    #                     :Convert high, low, and close to pandas Series;
    #                     :Resolve rolling min_periods;
    #                     :Compute rolling highest high and lowest low;
    #                     :Compute Williams %R;
    #                     :Return Williams %R series;
    #                     stop
    #                     @enduml
    # ***********************************************************************************************************************
    @staticmethod
    def compute_williams_r(high, low, close, window: int = 14, min_periods: int | None = None) -> pd.Series:
        validate_window(window, "window")

        h = to_series(high, name="high")
        l = to_series(low, name="low")
        c = to_series(close, name="close")

        mp = window if min_periods is None else min_periods

        highest_high = h.rolling(window=window, min_periods=mp).max()
        lowest_low = l.rolling(window=window, min_periods=mp).min()

        denom = (highest_high - lowest_low).replace(0.0, np.nan)
        # Negative value: 0 = overbought, -100 = oversold.
        wr = -100.0 * (highest_high - c) / denom
        wr.name = f"williams_r_{window}"
        return wr

    # ***********************************************************************************************************************
    # Functionname:       Oscillators.compute_stochrsi(series, window: int = 14, smooth1: int = 3,
    #                              smooth2: int = 3, fillna: bool = False)
    #
    # @brief              Compute StochRSI core value plus smoothed %K and %D.
    # @pre                window > 0, smooth1 > 0, smooth2 > 0
    # @post               Returns DataFrame with stochrsi, stochrsi_k, and stochrsi_d columns.
    # @param[in]          series: Input signal
    #                     window: RSI and StochRSI base window
    #                     smooth1: Smoothing window for %K
    #                     smooth2: Smoothing window for %D
    #                     fillna: Fill NaN/inf values with 0.0 when True
    # @param[out]         out: StochRSI metrics DataFrame
    #
    # @callsequence       @startuml
    #                     title Oscillators.compute_stochrsi
    #                     start
    #                     :Validate window, smooth1, and smooth2;
    #                     :Convert input to pandas Series;
    #                     :Compute RSI base series;
    #                     :Compute rolling RSI min and max;
    #                     :Compute stochrsi, stochrsi_k, and stochrsi_d;
    #                     if (fillna?) then (yes)
    #                       :Replace inf and NaN with 0.0;
    #                     endif
    #                     :Return StochRSI DataFrame;
    #                     stop
    #                     @enduml
    # ***********************************************************************************************************************
    @staticmethod
    def compute_stochrsi(series, window: int = 14, smooth1: int = 3, smooth2: int = 3, fillna: bool = False) -> pd.DataFrame:
        validate_window(window, "window")
        validate_window(smooth1, "smooth1")
        validate_window(smooth2, "smooth2")
        s = to_series(series)
        # RSI is computed from the imported module-level wrapper.
        rsi = compute_rsi(s, window=window, fillna=fillna)
        lowest_rsi  = rsi.rolling(window=window, min_periods=window).min()
        highest_rsi = rsi.rolling(window=window, min_periods=window).max()
        denom    = (highest_rsi - lowest_rsi).replace(0.0, np.nan)
        stochrsi = (rsi - lowest_rsi) / denom
        stochrsi_k = stochrsi.rolling(window=smooth1, min_periods=smooth1).mean()
        stochrsi_d = stochrsi_k.rolling(window=smooth2, min_periods=smooth2).mean()
        result = pd.DataFrame({"stochrsi": stochrsi, "stochrsi_k": stochrsi_k, "stochrsi_d": stochrsi_d}, index=s.index)
        if fillna:
            result = result.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return result

    # ***********************************************************************************************************************
    # Functionname:       Oscillators.compute_stochrsi_k(series, window: int = 14, smooth1: int = 3,
    #                              smooth2: int = 3, fillna: bool = False)
    #
    # @brief              Return StochRSI %K series.
    # @pre                Same preconditions as compute_stochrsi.
    # @post               Returns stochrsi_k series.
    # @param[in]          series: Input signal
    #                     window: RSI and StochRSI base window
    #                     smooth1: Smoothing window for %K
    #                     smooth2: Smoothing window for %D
    #                     fillna: Fill NaN/inf values with 0.0 when True
    # @param[out]         out: StochRSI %K series
    #
    # @callsequence       @startuml
    #                     title Oscillators.compute_stochrsi_k
    #                     start
    #                     :Call compute_stochrsi;
    #                     :Extract stochrsi_k column;
    #                     :Return StochRSI %K series;
    #                     stop
    #                     @enduml
    # ***********************************************************************************************************************
    @staticmethod
    def compute_stochrsi_k(series,  window: int = 14, smooth1: int = 3, smooth2: int = 3, fillna: bool = False) -> pd.Series:
        return Oscillators.compute_stochrsi(series=series, window=window, smooth1=smooth1, smooth2=smooth2, fillna=fillna)["stochrsi_k"]

    # ***********************************************************************************************************************
    # Functionname:       Oscillators.compute_stochrsi_d(series, window: int = 14, smooth1: int = 3,
    #                              smooth2: int = 3, fillna: bool = False)
    #
    # @brief              Return StochRSI %D series.
    # @pre                Same preconditions as compute_stochrsi.
    # @post               Returns stochrsi_d series.
    # @param[in]          series: Input signal
    #                     window: RSI and StochRSI base window
    #                     smooth1: Smoothing window for %K
    #                     smooth2: Smoothing window for %D
    #                     fillna: Fill NaN/inf values with 0.0 when True
    # @param[out]         out: StochRSI %D series
    #
    # @callsequence       @startuml
    #                     title Oscillators.compute_stochrsi_d
    #                     start
    #                     :Call compute_stochrsi;
    #                     :Extract stochrsi_d column;
    #                     :Return StochRSI %D series;
    #                     stop
    #                     @enduml
    # ***********************************************************************************************************************
    @staticmethod
    def compute_stochrsi_d(series, window: int = 14, smooth1: int = 3, smooth2: int = 3, fillna: bool = False) -> pd.Series:
        return Oscillators.compute_stochrsi(series=series, window=window, smooth1=smooth1, smooth2=smooth2, fillna=fillna)["stochrsi_d"]

# ---------------------------------------------------------------------------
# Backward-compatible module-level wrappers
# ---------------------------------------------------------------------------
def compute_stochastic(high, low, close, window: int = 14, smooth_window: int = 3, min_periods: int | None = None) -> pd.DataFrame:
    return Oscillators.compute_stochastic(high=high, low=low, close=close, window=window, smooth_window=smooth_window, min_periods=min_periods)

def compute_stochastic_signal(high, low, close, window: int = 14, smooth_window: int = 3, min_periods: int | None = None) -> pd.Series:
    return Oscillators.compute_stochastic_signal(high=high, low=low, close=close, window=window, smooth_window=smooth_window, min_periods=min_periods)

def compute_williams_r(high, low, close, window: int = 14, min_periods: int | None = None) -> pd.Series:
    return Oscillators.compute_williams_r(high=high, low=low, close=close, window=window, min_periods=min_periods)

def compute_stochrsi(series, window: int = 14, smooth1: int = 3, smooth2: int = 3, fillna: bool = False) -> pd.DataFrame:
    return Oscillators.compute_stochrsi(series=series, window=window, smooth1=smooth1, smooth2=smooth2, fillna=fillna)

def compute_stochrsi_k(series, window: int = 14, smooth1: int = 3, smooth2: int = 3, fillna: bool = False) -> pd.Series:
    return Oscillators.compute_stochrsi_k(series=series, window=window, smooth1=smooth1, smooth2=smooth2, fillna=fillna)

def compute_stochrsi_d(series, window: int = 14, smooth1: int = 3, smooth2: int = 3, fillna: bool = False) -> pd.Series:
    return Oscillators.compute_stochrsi_d(series=series, window=window, smooth1=smooth1, smooth2=smooth2, fillna=fillna)

__all__ = ["Oscillators", "compute_stochastic", "compute_stochastic_signal", "compute_williams_r", "compute_stochrsi", 
           "compute_stochrsi_k", "compute_stochrsi_d"]
