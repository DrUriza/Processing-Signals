from __future__ import annotations
import pandas   as pd

# \file **********************************************************************
# COMPANY:            ELATIN
# PROJECT:            SIGNAL_ANALYSIS
# COMPONENT:          UTILS
# MODULE NAME:        helpers.py
# DESCRIPTION:        @brief Shared scalar-series helpers and window validation
# CREATION DATE:      26.04.2026
# VERSION:            $Revision: 0.1$
# CHANGES:            26.04.2026 - Added Doxygen-style function documentation.
# *****************************************************************************

# *******************************************************************************************************************
# Functionname:       to_series(data, name: str = "series")
#
# @brief              Convert input data to a pandas Series of floats.
# @pre                data must be array-like or a pandas Series.
# @post               Returns independent float-valued pandas Series.
# @param[in]          data: Input array-like or pandas Series
#                     name: Output series name when input is not already a Series
# @param[out]         out: Float-valued pandas Series copy
#
# @callsequence       @startuml
#                     title to_series
#                     start
#                     if (data is pandas Series?) then (yes)
#                       :Cast to float and copy;
#                       :Return Series;
#                       stop
#                     endif
#                     :Build pandas Series with dtype float and provided name;
#                     :Return Series;
#                     stop
#                     @enduml
# *******************************************************************************************************************
def to_series(data, name: str = "series") -> pd.Series:
    if isinstance(data, pd.Series):
        return data.astype(float).copy()
    return pd.Series(data, dtype=float, name=name)


# *******************************************************************************************************************
# Functionname:       validate_window(window: int, name: str = "window")
#
# @brief              Validate that a window-like parameter is a positive integer.
# @pre                name should be a valid parameter label.
# @post               Raises ValueError when window is invalid.
# @param[in]          window: Candidate window-like value
#                     name: Parameter name used in error messages
# @param[out]         out: None
#
# @callsequence       @startuml
#                     title validate_window
#                     start
#                     if (window is not positive integer?) then (yes)
#                       :Raise ValueError;
#                       stop
#                     endif
#                     :Return without error;
#                     stop
#                     @enduml
# *******************************************************************************************************************
def validate_window(window: int, name: str = "window") -> None:
    if not isinstance(window, int) or window <= 0:
        raise ValueError(f"{name} must be a positive integer.")


# *******************************************************************************************************************
# Functionname:       ema_series(series: pd.Series, window: int, adjust: bool = False,
#                              min_periods: int | None = None)
#
# @brief              Reusable EMA helper.
# @pre                window must be a positive integer.
# @post               Returns exponential moving average Series.
# @param[in]          series: Input pandas Series
#                     window: EMA span parameter
#                     adjust: Pandas EWM adjust flag
#                     min_periods: Minimum periods before returning valid values
# @param[out]         out: Exponential moving average Series
#
# @callsequence       @startuml
#                     title ema_series
#                     start
#                       :Call validate_window(window);
#                       :Resolve effective min_periods;
#                       :Compute EWM mean using pandas;
#                       :Return EMA series;
#                     stop
#                     @enduml
# *******************************************************************************************************************
def ema_series(series: pd.Series, window: int, adjust: bool = False, min_periods: int | None = None) -> pd.Series:
    validate_window(window, "window")
    mp = window if min_periods is None else min_periods
    return series.ewm(span=window, adjust=adjust, min_periods=mp).mean()
