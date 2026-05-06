from __future__ import annotations

# \file **********************************************************************
# COMPANY:            ELATIN
# PROJECT:            SIGNAL_ANALYSIS
# COMPONENT:          UTILS
# MODULE NAME:        ohlc.py
# DESCRIPTION:        @brief OHLC utility helpers shared across indicators
# CREATION DATE:      26.04.2026
# VERSION:            $Revision: 0.2$
# CHANGES:            26.04.2026 - Aligned header and indentation style.
# *****************************************************************************

import pandas as pd

from signal_analysis.utils.helpers import to_series


# ***********************************************************************************************************************
# Functionname:       true_range(high, low, close) -> pd.Series
#
# @brief              Compute True Range: max(H-L, |H-prev_C|, |L-prev_C|).
# @pre                high, low, close must be array-like with matching length.
# @post               Returns a pd.Series named "true_range".
# @param[in]          high: High price series
#                     low: Low price series
#                     close: Close price series
# @param[out]         out: True Range series
#
# @callsequence       @startuml
#                     title true_range
#                     start
#                     :Convert high, low, and close to pandas Series;
#                     :Shift close series by one bar;
#                     :Compute three True Range components;
#                     :Take row-wise maximum;
#                     :Rename output to true_range;
#                     :Return True Range series;
#                     stop
#                     @enduml
# ***********************************************************************************************************************
def true_range(high, low, close) -> pd.Series:
    h = to_series(high,  name="high")
    l = to_series(low,   name="low")
    c = to_series(close, name="close")

    prev_close = c.shift(1)

    tr = pd.concat([(h - l).abs(),
                    (h - prev_close).abs(),
                    (l - prev_close).abs()],
                   axis=1).max(axis=1)
    tr.name = "true_range"
    return tr


# ***********************************************************************************************************************
# Functionname:       typical_price(high, low, close) -> pd.Series
#
# @brief              Compute Typical Price: (H + L + C) / 3.
# @pre                high, low, close must be array-like with matching length.
# @post               Returns a pd.Series named "typical_price".
# @param[in]          high: High price series
#                     low: Low price series
#                     close: Close price series
# @param[out]         out: Typical Price series
#
# @callsequence       @startuml
#                     title typical_price
#                     start
#                     :Convert high, low, and close to pandas Series;
#                     :Compute (H + L + C) / 3.0;
#                     :Rename output to typical_price;
#                     :Return Typical Price series;
#                     stop
#                     @enduml
# ***********************************************************************************************************************
def typical_price(high, low, close) -> pd.Series:
    h = to_series(high,  name="high")
    l = to_series(low,   name="low")
    c = to_series(close, name="close")

    tp = (h + l + c) / 3.0
    tp.name = "typical_price"
    return tp


__all__ = ["true_range", "typical_price"]
