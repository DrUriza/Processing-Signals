from __future__ import annotations

import numpy as np
import pandas as pd

from signal_analysis.utils.helpers import validate_window


# \file **********************************************************************
# COMPANY:            ELATIN
# PROJECT:            SIGNAL_ANALYSIS
# COMPONENT:          SIGNALS
# MODULE NAME:        helpers.py
# DESCRIPTION:        @brief Signal helper utilities
# CREATION DATE:      30.04.2026
# VERSION:            $Revision: 0.1$
# CHANGES:            30.04.2026 - Added signal activation helper functions.
# *****************************************************************************


# ***********************************************************************************************************************
# Functionname:       detectar_compra(signal_array)
#
# @brief              Detect rising edges in a binary signal array (0->1 transitions).
# @pre                signal_array must be array-like with values convertible to float.
# @post               Returns float array of 1.0 at activation points, 0.0 elsewhere.
# @param[in]          signal_array: Input binary signal (0/1 values)
# @param[out]         out: Numpy float array of buy activation events
#
# @callsequence       @startuml
#                     title detectar_compra
#                     start
#                       :Convert input to pandas Series (float, fill NaN=0);
#                       :Shift series by 1 to get previous values;
#                       :Detect positions where current=1 and previous=0;
#                       :Return float numpy array;
#                     stop
#                     @enduml
# ***********************************************************************************************************************
def detectar_compra(signal_array) -> np.ndarray:
    signal_series = pd.Series(signal_array, dtype=float).fillna(0.0)
    prev = signal_series.shift(1).fillna(0.0)
    return ((signal_series == 1.0) & (prev == 0.0)).astype(float).to_numpy(dtype=float)


# ***********************************************************************************************************************
# Functionname:       detectar_venta(signal_array)
#
# @brief              Detect rising edges in a binary signal and encode them as -1.0 sell activations.
# @pre                signal_array must be array-like with values convertible to float.
# @post               Returns float array of -1.0 at sell activation points, 0.0 elsewhere.
# @param[in]          signal_array: Input binary signal (0/1 values)
# @param[out]         out: Numpy float array of sell activation events
#
# @callsequence       @startuml
#                     title detectar_venta
#                     start
#                       :Convert input to pandas Series (float, fill NaN=0);
#                       :Shift series by 1 to get previous values;
#                       :Detect positions where current=1 and previous=0;
#                       :Assign -1.0 at activation points, 0.0 elsewhere;
#                       :Return float numpy array;
#                     stop
#                     @enduml
# ***********************************************************************************************************************
def detectar_venta(signal_array) -> np.ndarray:
    signal_series = pd.Series(signal_array, dtype=float).fillna(0.0)
    prev = signal_series.shift(1).fillna(0.0)
    activacion = (signal_series == 1.0) & (prev == 0.0)
    return np.where(activacion, -1.0, 0.0).astype(float)


# ***********************************************************************************************************************
# Functionname:       high_volume(data: pd.DataFrame, lookback: int = 500, n_last: int = 5, k: float = 1.2)
#
# @brief              Detect abnormally high volume based on rolling mean+k*std threshold.
# @pre                data must contain a 'volume' column; lookback > 0; n_last > 0; k >= 0.
# @post               Returns 1 if recent mean volume exceeds base threshold, 0 otherwise.
# @param[in]          data: Input DataFrame with 'volume' column
#                     lookback: Rolling window length for baseline statistics
#                     n_last: Number of most recent bars to compute current mean
#                     k: Standard deviation multiplier for the threshold
# @param[out]         out: Integer 1 (high volume) or 0 (normal)
#
# @callsequence       @startuml
#                     title high_volume
#                     start
#                     :Validate lookback, n_last, k, and 'volume' column;
#                     if (insufficient data?) then (yes)
#                       :Return 0;
#                       stop
#                     endif
#                     :Compute mean of last n_last volume bars;
#                     :Compute rolling mean and std over lookback window;
#                     if (baseline stats are NaN?) then (yes)
#                       :Return 0;
#                       stop
#                     endif
#                     if (cur_mean > base_mean + k * base_std?) then (yes)
#                       :Return 1;
#                     else
#                       :Return 0;
#                     endif
#                     stop
#                     @enduml
# ***********************************************************************************************************************
def high_volume(data: pd.DataFrame, lookback: int = 500, n_last: int = 5, k: float = 1.2) -> int:
    validate_window(lookback, "lookback")
    validate_window(n_last, "n_last")
    if k < 0:
        raise ValueError("k must be >= 0")
    if "volume" not in data.columns:
        raise ValueError("column 'volume' not found in DataFrame")

    v = pd.Series(data["volume"], dtype=float)
    if len(v) < max(lookback, n_last):
        return 0

    cur_mean = float(v.iloc[-n_last:].mean())
    base_mean = float(v.rolling(lookback).mean().iloc[-1])
    base_std = float(v.rolling(lookback).std().iloc[-1])

    if np.isnan(base_mean) or np.isnan(base_std):
        return 0
    return int(cur_mean > base_mean + k * base_std)


__all__ = ["detectar_compra", "detectar_venta", "high_volume"]
