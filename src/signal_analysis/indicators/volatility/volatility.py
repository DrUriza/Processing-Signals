from   __future__                  import annotations
import numpy                       as np
import pandas                      as pd
from signal_analysis.utils.helpers import to_series, validate_window
from signal_analysis.utils.ohlc    import true_range
from signal_analysis.indicators.trend.trend_strength import compute_directional_indicators

# \file **********************************************************************
# COMPANY:            ELATIN
# PROJECT:            SIGNAL_ANALYSIS
# COMPONENT:          INDICATORS
# MODULE NAME:        volatility.py
# DESCRIPTION:        @brief Volatility indicator helpers
# CREATION DATE:      22.04.2026
# VERSION:            $Revision: 0.2$
# CHANGES:            22.04.2026 - Migrated to short banner comments.
#                     22.04.2026 - Refactored to class-based OOP.
# *****************************************************************************

class VolatilityIndicators:
    # ***********************************************************************************************************************
    # Functionname:       VolatilityIndicators.compute_bollinger_bands(series, window: int = 20, n_std: float = 2.0,
    #                              min_periods: int | None = None)
    #
    # @brief              Compute Bollinger Bands and derived metrics from a single series.
    # @pre                window > 0
    # @post               Returns DataFrame with middle, upper, lower, bandwidth, and percent_b columns.
    # @param[in]          series: Input signal
    #                     window: Rolling window length
    #                     n_std: Standard deviation multiplier
    #                     min_periods: Minimum periods for rolling operations
    # @param[out]         out: Bollinger metrics DataFrame
    #
    # @callsequence       @startuml
    #                     title VolatilityIndicators.compute_bollinger_bands
    #                     start
    #                     :Validate window;
    #                     :Convert input to pandas Series;
    #                     :Resolve rolling min_periods;
    #                     :Compute rolling mean and standard deviation;
    #                     :Compute upper and lower bands;
    #                     :Compute bandwidth and percent_b;
    #                     :Return Bollinger metrics DataFrame;
    #                     stop
    #                     @enduml
    # ***********************************************************************************************************************
    @staticmethod
    def compute_bollinger_bands(series, window: int = 20, n_std: float = 2.0, min_periods: int | None = None) -> pd.DataFrame:
        validate_window(window, "window")
        s = to_series(series)
        mp = window if min_periods is None else min_periods

        middle = s.rolling(window=window, min_periods=mp).mean()
        rolling_std = s.rolling(window=window, min_periods=mp).std(ddof=0)

        # Symmetric bands centered on rolling mean.
        upper = middle + n_std * rolling_std
        lower = middle - n_std * rolling_std
        bandwidth = upper - lower

        percent_b = (s - lower) / (upper - lower)
        percent_b = percent_b.replace([np.inf, -np.inf], np.nan)

        result = pd.DataFrame({"bb_middle": middle,
                               "bb_upper": upper,
                               "bb_lower": lower,
                               "bb_bandwidth": bandwidth,
                               "bb_percent_b": percent_b}, index=s.index)
        return result

    # ***********************************************************************************************************************
    # Functionname:       VolatilityIndicators.compute_tr(high, low, close)
    #
    # @brief              Provide true range under indicator namespace naming.
    # @pre                Inputs must be aligned high/low/close sequences.
    # @post               Returns true range series named true_range.
    # @param[in]          high: High price series
    #                     low: Low price series
    #                     close: Close price series
    # @param[out]         out: True range as pandas Series
    #
    # @callsequence       @startuml
    #                     title VolatilityIndicators.compute_tr
    #                     start
    #                     :Call true_range helper;
    #                     :Rename output series to true_range;
    #                     :Return true range series;
    #                     stop
    #                     @enduml
    # ***********************************************************************************************************************
    @staticmethod
    def compute_tr(high, low, close) -> pd.Series:
        tr = true_range(high, low, close)
        tr.name = "true_range"
        return tr

    # ***********************************************************************************************************************
    # Functionname:       VolatilityIndicators.compute_atr(high, low, close, window: int = 14,
    #                              min_periods: int | None = None)
    #
    # @brief              Compute Average True Range (ATR) with Wilder-style smoothing.
    # @pre                window > 0 and inputs must be aligned high/low/close sequences.
    # @post               Returns ATR series named atr_window.
    # @param[in]          high: High price series
    #                     low: Low price series
    #                     close: Close price series
    #                     window: ATR period
    #                     min_periods: Minimum periods for EMA initialization
    # @param[out]         out: ATR as pandas Series
    #
    # @callsequence       @startuml
    #                     title VolatilityIndicators.compute_atr
    #                     start
    #                     :Validate window;
    #                     :Convert high, low, and close to pandas Series;
    #                     :Compute true range;
    #                     :Resolve EMA min_periods;
    #                     :Apply Wilder EMA smoothing;
    #                     :Rename output series to atr_window;
    #                     :Return ATR series;
    #                     stop
    #                     @enduml
    # ***********************************************************************************************************************
    @staticmethod
    def compute_atr(high, low, close, window: int = 14, min_periods: int | None = None) -> pd.Series:
        validate_window(window, "window")
        h = to_series(high, name="high")
        l = to_series(low, name="low")
        c = to_series(close, name="close")

        tr  = true_range(h, l, c)
        mp  = window if min_periods is None else min_periods
        # Wilder EMA smoothing (alpha = 1/window).
        atr = tr.ewm(alpha=1 / window, adjust=False, min_periods=mp).mean()
        atr.name = f"atr_{window}"
        return atr

    # ***********************************************************************************************************************
    # Functionname:       VolatilityIndicators.compute_adx_directional_signal(high, low, close, window: int = 14)
    #
    # @brief              Compute ADX and directional movement proxy with crossover position changes.
    # @pre                window > 0 and inputs must be aligned high/low/close sequences.
    # @post               Returns DataFrame with adx, md_plus, md_minus, and md_position columns.
    # @param[in]          high: High price series
    #                     low: Low price series
    #                     close: Close price series
    #                     window: ADX period
    # @param[out]         out: ADX directional metrics and transition signal
    #
    # @callsequence       @startuml
    #                     title VolatilityIndicators.compute_adx_directional_signal
    #                     start
    #                     :Compute directional indicators via TrendStrength;
    #                     :Extract md_plus and md_minus (+DI/-DI);
    #                     :Extract ADX series;
    #                     :Compute md_position as crossover transition (diff);
    #                     :Return DataFrame with adx, md_plus, md_minus, md_position;
    #                     stop
    #                     @enduml
    # ***********************************************************************************************************************
    @staticmethod
    def compute_adx_directional_signal(high, low, close, window: int = 14) -> pd.DataFrame:
        di = compute_directional_indicators(high, low, close, window=window)

        md_plus = di["plus_di"].rename("md_plus")
        md_minus = di["minus_di"].rename("md_minus")
        adx = di["adx"].rename("adx")

        md_signal = np.where(md_plus > md_minus, 1.0, 0.0)
        md_position = pd.Series(md_signal, index=di.index, name="md_position").diff().fillna(0.0)

        return pd.DataFrame(
            {
                "adx": adx,
                "md_plus": md_plus,
                "md_minus": md_minus,
                "md_position": md_position,
            },
            index=di.index,
        )

    # ***********************************************************************************************************************
    # Functionname:       VolatilityIndicators.compute_bollinger_reference_signal(series, reference,
    #                              window: int | None = None, n_std: float = 2.0,
    #                              min_window: int = 5)
    #
    # @brief              Build Bollinger bands with dynamic default window and reference-based crossover position.
    # @pre                min_window > 0 and reference length must match input series length.
    # @post               Returns Bollinger metrics plus bb_signal and bb_position columns.
    # @param[in]          series: Input price series (typically close)
    #                     reference: Comparison series (typically prediction)
    #                     window: Optional explicit Bollinger window
    #                     n_std: Standard deviation multiplier
    #                     min_window: Lower bound for dynamic window when window is None
    # @param[out]         out: DataFrame with Bollinger metrics and signal/position
    #
    # @callsequence       @startuml
    #                     title VolatilityIndicators.compute_bollinger_reference_signal
    #                     start
    #                     :Validate min_window;
    #                     :Convert series to pandas Series;
    #                     :Align reference to series index;
    #                     if (window is None?) then (yes)
    #                       :Compute dynamic window from series length;
    #                     endif
    #                     :Compute Bollinger Bands;
    #                     :Compute bb_signal and bb_position crossover;
    #                     :Return concatenated Bollinger metrics with signals;
    #                     stop
    #                     @enduml
    # ***********************************************************************************************************************
    @staticmethod
    def compute_bollinger_reference_signal(
        series,
        reference,
        window: int | None = None,
        n_std: float = 2.0,
        min_window: int = 5,
    ) -> pd.DataFrame:
        validate_window(min_window, "min_window")
        s = to_series(series, name="signal")

        if isinstance(reference, pd.Series):
            ref = reference.reindex(s.index)
        else:
            ref = to_series(reference, name="reference")
            if len(ref) != len(s):
                raise ValueError("reference must have the same length as series")
            ref.index = s.index

        if window is None:
            window = max(min_window, len(s) // 10)

        bb = VolatilityIndicators.compute_bollinger_bands(s, window=window, n_std=n_std)

        bb_signal = np.where(bb["bb_middle"] > ref, 1.0, 0.0)
        bb_signal_series = pd.Series(bb_signal, index=s.index, name="bb_signal")
        bb_position = bb_signal_series.diff().fillna(0.0).rename("bb_position")

        return pd.concat([bb, bb_signal_series, bb_position], axis=1)

# ---------------------------------------------------------------------------
# Backward-compatible module-level wrappers
# ---------------------------------------------------------------------------
def compute_bollinger_bands(series, window: int = 20, n_std: float = 2.0, min_periods: int | None = None) -> pd.DataFrame:
    return VolatilityIndicators.compute_bollinger_bands(series, window=window, n_std=n_std, min_periods=min_periods)

def compute_tr(high, low, close) -> pd.Series:
    return VolatilityIndicators.compute_tr(high, low, close)

def compute_atr(high, low, close, window: int = 14, min_periods: int | None = None) -> pd.Series:
    return VolatilityIndicators.compute_atr(high, low, close, window=window, min_periods=min_periods)

def compute_adx_directional_signal(high, low, close, window: int = 14) -> pd.DataFrame:
    return VolatilityIndicators.compute_adx_directional_signal(high, low, close, window=window)

def compute_bollinger_reference_signal(
    series,
    reference,
    window: int | None = None,
    n_std: float = 2.0,
    min_window: int = 5,
) -> pd.DataFrame:
    return VolatilityIndicators.compute_bollinger_reference_signal(
        series=series,
        reference=reference,
        window=window,
        n_std=n_std,
        min_window=min_window,
    )

__all__ = [
    "VolatilityIndicators",
    "compute_bollinger_bands",
    "compute_tr",
    "compute_atr",
    "compute_adx_directional_signal",
    "compute_bollinger_reference_signal",
]
