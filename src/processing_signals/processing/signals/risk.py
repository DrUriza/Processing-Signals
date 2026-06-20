from __future__ import annotations

import pandas as pd


# \file **********************************************************************
# COMPANY:            ELATIN
# PROJECT:            SIGNAL_ANALYSIS
# COMPONENT:          SIGNALS
# MODULE NAME:        risk.py
# DESCRIPTION:        @brief Dynamic stop-loss and take-profit helpers
# CREATION DATE:      30.04.2026
# VERSION:            $Revision: 0.1$
# CHANGES:            30.04.2026 - Added ATR/ADX directional risk helper.
# *****************************************************************************


# ***********************************************************************************************************************
# Functionname:       stop_loss_take_profit(data: pd.DataFrame, close_col: str = "close",
#                              atr_col: str = "ATR", adx_col: str = "ADX",
#                              md_plus_col: str = "MD+", md_minus_col: str = "MD-")
#
# @brief              Compute dynamic stop-loss and take-profit levels using ATR, ADX, and directional bias.
# @pre                data must contain all required columns; ATR must be > 0; data must not be empty.
# @post               Returns (stop_loss, take_profit) as a float tuple based on directional strength.
# @param[in]          data: DataFrame with indicator columns
#                     close_col: Close price column name
#                     atr_col: ATR column name
#                     adx_col: ADX column name
#                     md_plus_col: +DI column name
#                     md_minus_col: -DI column name
# @param[out]         out: Tuple (stop_loss, take_profit) as floats
#
# @callsequence       @startuml
#                     title stop_loss_take_profit
#                     start
#                     :Validate required columns and non-empty data;
#                     :Extract last values for close, ATR, ADX, MD+, MD-;
#                     if (ATR <= 0?) then (yes)
#                       :Raise ValueError;
#                       stop
#                     endif
#                     :Compute directional strength ratios;
#                     :Set sl_distance = 1.5 * ATR;
#                     if (ADX > 25?) then (yes)
#                       :Set rr_ratio = 2.0;
#                     else
#                       :Set rr_ratio = 1.2;
#                     endif
#                     if (MD+ > MD-?) then (yes)
#                       :Compute bullish SL below close and TP above;
#                     elseif (MD- > MD+?) then (yes)
#                       :Compute bearish SL above close and TP below;
#                     else
#                       :Use symmetric ATR-based SL and TP;
#                     endif
#                     :Return (stop_loss, take_profit);
#                     stop
#                     @enduml
# ***********************************************************************************************************************
def stop_loss_take_profit(
    data: pd.DataFrame,
    close_col: str = "close",
    atr_col: str = "ATR",
    adx_col: str = "ADX",
    md_plus_col: str = "MD+",
    md_minus_col: str = "MD-",
) -> tuple[float, float]:
    required = [close_col, atr_col, adx_col, md_plus_col, md_minus_col]
    missing = [col for col in required if col not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if data.empty:
        raise ValueError("data must not be empty")

    close = float(data[close_col].iloc[-1])
    atr = float(data[atr_col].iloc[-1])
    adx = float(data[adx_col].iloc[-1])
    mdp = float(data[md_plus_col].iloc[-1])
    mdm = float(data[md_minus_col].iloc[-1])

    if atr <= 0:
        raise ValueError("ATR must be > 0 for stop-loss/take-profit computation")

    if (mdp + mdm) != 0:
        strength_up = mdp / (mdp + mdm)
        strength_down = mdm / (mdp + mdm)
    else:
        strength_up = strength_down = 0.5

    sl_distance = 1.5 * atr
    rr_ratio = 2.0 if adx > 25 else 1.2

    if mdp > mdm:
        stop_loss = close - sl_distance
        take_profit = close + sl_distance * rr_ratio * (1 + strength_up / 2)
    elif mdp < mdm:
        stop_loss = close + sl_distance
        take_profit = close - sl_distance * rr_ratio * (1 + strength_down / 2)
    else:
        stop_loss = close - atr
        take_profit = close + atr

    return float(stop_loss), float(take_profit)


__all__ = ["stop_loss_take_profit"]
