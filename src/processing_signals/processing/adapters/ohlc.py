from __future__                    import annotations
import pandas                      as pd
from signal_analysis.utils.helpers import to_series, validate_window
from signal_analysis.utils.ohlc    import typical_price

# \file **********************************************************************
# COMPANY:            ELATIN
# PROJECT:            SIGNAL_ANALYSIS
# COMPONENT:          ADAPTERS
# MODULE NAME:        ohlc
# DESCRIPTION:        @brief OHLC adapter utilities. Bidirectional conversion
#                     between scalar series and OHLC DataFrames.
# CREATION DATE:      22.04.2026
# VERSION:            $Revision: 0.2$
# CHANGES:            22.04.2026 - Migrated to short banner comments.
#                     22.04.2026 - Refactored to class-based OOP.
# *****************************************************************************

class OHLCAdapter:
    # *******************************************************************************************************************
    # Functionname:   series_to_ohlc_windows(series, window, step) -> pd.DataFrame
    #
    # @brief          Convert a 1-D scalar series into windowed OHLC summaries.
    # @pre            series is array-like; window > 0
    # @post           Returns DataFrame with open, high, low, close, start_idx, end_idx
    # @param[in]      series: Input scalar time series
    #                 window: Number of samples per window
    #                 step:   Step between windows (defaults to window)
    # @param[out]     result: OHLC DataFrame
    #
    # @callsequence       @startuml
    #                     title OHLCAdapter.series_to_ohlc_windows
    #                     start
    #                     :Validate window;
    #                     :Convert input to pd.Series;
    #                     if (step is None?) then (yes)
    #                       :Set step = window;
    #                     endif
    #                     :Validate step;
    #                     repeat
    #                       :Slice chunk [start:start+window];
    #                       :Compute open/high/low/close;
    #                       :Append row with start/end indices;
    #                     repeat while (more windows)
    #                     :Build and return DataFrame;
    #                     end
    #                     @enduml
    # *******************************************************************************************************************
    @staticmethod
    def series_to_ohlc_windows(series, window: int, step: int | None = None) -> pd.DataFrame:
        validate_window(window, "window")
        s = to_series(series)
        if step is None:
            step = window
        validate_window(step, "step")
        rows = []
        for start in range(0, len(s) - window + 1, step):
            chunk = s.iloc[start : start + window]
            rows.append({"open":      float(chunk.iloc[0]),
                         "high":      float(chunk.max()),
                         "low":       float(chunk.min()),
                         "close":     float(chunk.iloc[-1]),
                         "start_idx": int(start),
                         "end_idx":   int(start + window - 1)})
        return pd.DataFrame(rows)

    # *******************************************************************************************************************
    # Functionname:   series_to_ohlc_dataframe(series, window, step, index_mode) -> pd.DataFrame
    #
    # @brief          Convert a scalar series into an indexed OHLC DataFrame.
    # @pre            series is array-like; window > 0; index_mode in {"end","start","range"}
    # @post           Returns OHLC DataFrame indexed per index_mode
    # @param[in]      series:     Input scalar time series
    #                 window:     Number of samples per window
    #                 step:       Step between windows (defaults to window)
    #                 index_mode: "end" | "start" | "range"
    # @param[out]     result: Indexed OHLC DataFrame
    #
    # @callsequence       @startuml
    #                     title OHLCAdapter.series_to_ohlc_dataframe
    #                     start
    #                     :Call series_to_ohlc_windows;
    #                     if (index_mode == "end"?) then (yes)
    #                       :Set index = end_idx;
    #                       :Return DataFrame;
    #                     elseif (index_mode == "start"?) then (yes)
    #                       :Set index = start_idx;
    #                       :Return DataFrame;
    #                     elseif (index_mode == "range"?) then (yes)
    #                       :Return DataFrame unchanged;
    #                     else (no)
    #                       :Raise ValueError;
    #                     endif
    #                     end
    #                     @enduml
    # *******************************************************************************************************************
    @staticmethod
    def series_to_ohlc_dataframe(series, window: int, step: int | None = None, index_mode: str = "end") -> pd.DataFrame:
        ohlc = OHLCAdapter.series_to_ohlc_windows(series=series, window=window, step=step)
        if index_mode == "end":
            return ohlc.set_index("end_idx", drop=False)
        elif index_mode == "start":
            return ohlc.set_index("start_idx", drop=False)
        elif index_mode == "range":
            return ohlc
        else:
            raise ValueError("index_mode must be one of: 'end', 'start', 'range'.")

    # *******************************************************************************************************************
    # Functionname:   ohlc_to_series(ohlc, mode) -> pd.Series
    #
    # @brief          Reduce an OHLC DataFrame to a single price series.
    # @pre            ohlc is a DataFrame with columns: open, high, low, close
    # @post           Returns a pd.Series of the selected price representation
    # @param[in]      ohlc: OHLC DataFrame (must contain open, high, low, close)
    #                 mode: Price representation —
    #                       "close"    → close price  (default)
    #                       "typical"  → (H + L + C) / 3
    #                       "weighted" → (H + L + 2·C) / 4
    # @param[out]     result: pd.Series
    #
    # @callsequence       @startuml
    #                     title OHLCAdapter.ohlc_to_series
    #                     start
    #                     :Validate required OHLC columns;
    #                     if (mode == "close"?) then (yes)
    #                       :Return close column copy;
    #                     elseif (mode == "typical"?) then (yes)
    #                       :Compute (H + L + C) / 3;
    #                       :Return typical price series;
    #                     elseif (mode == "weighted"?) then (yes)
    #                       :Compute (H + L + 2C) / 4;
    #                       :Return weighted close series;
    #                     else (no)
    #                       :Raise ValueError;
    #                     endif
    #                     end
    #                     @enduml
    # *******************************************************************************************************************
    @staticmethod
    def ohlc_to_series(ohlc: pd.DataFrame, mode: str = "close") -> pd.Series:
        required = {"open", "high", "low", "close"}
        if not required.issubset(ohlc.columns):
            raise ValueError(f"ohlc must contain columns: {required}")
        if mode == "close":
            result = ohlc["close"].copy()
            result.name = "close"
        elif mode == "typical":
            result = typical_price(ohlc["high"], ohlc["low"], ohlc["close"])
        elif mode == "weighted":
            result = (ohlc["high"] + ohlc["low"] + 2.0 * ohlc["close"]) / 4.0
            result.name = "weighted_close"
        else:
            raise ValueError("mode must be one of: 'close', 'typical', 'weighted'.")
        return result

# ----------------------------------------------------------------------------------------------------------------------
# Backward-compatible module-level wrappers
# ----------------------------------------------------------------------------------------------------------------------
def series_to_ohlc_windows(series, window: int, step: int | None = None) -> pd.DataFrame:
    return OHLCAdapter.series_to_ohlc_windows(series=series, window=window, step=step)

def series_to_ohlc_dataframe(series, window: int, step: int | None = None, index_mode: str = "end") -> pd.DataFrame:
    return OHLCAdapter.series_to_ohlc_dataframe(series=series, window=window, step=step, index_mode=index_mode)

def ohlc_to_series(ohlc: pd.DataFrame, mode: str = "close") -> pd.Series:
    return OHLCAdapter.ohlc_to_series(ohlc=ohlc, mode=mode)

__all__ = ["OHLCAdapter", "series_to_ohlc_windows", "series_to_ohlc_dataframe", "ohlc_to_series"]

