from __future__ import annotations

import numpy as np
import pandas as pd

from signal_analysis.utils.helpers import ema_series, to_series, validate_window
from signal_analysis.utils.ohlc import true_range


# \file **********************************************************************
# COMPANY:            ELATIN
# PROJECT:            SIGNAL_ANALYSIS
# COMPONENT:          INDICATORS
# MODULE NAME:        trend_strength.py
# DESCRIPTION:        @brief Trend strength indicators (MACD, DI, ADX)
# CREATION DATE:      22.04.2026
# VERSION:            $Revision: 0.2$
# CHANGES:            22.04.2026 - Migrated to short banner comments.
#                     22.04.2026 - Refactored to class-based OOP.
# *****************************************************************************


class TrendStrength:

    # ***********************************************************************************************************************
    # Functionname:       TrendStrength.compute_macd_components(series, window_slow: int = 26,
    #                              window_fast: int = 12, window_signal: int = 9,
    #                              min_periods: int | None = None)
    #
    # @brief              Compute MACD line, signal line, and histogram.
    # @pre                windows > 0 and window_fast < window_slow
    # @post               Returns DataFrame with macd, macd_signal, and macd_hist.
    # @param[in]          series: Input signal
    #                     window_slow: Slow EMA window
    #                     window_fast: Fast EMA window
    #                     window_signal: Signal EMA window
    #                     min_periods: Minimum periods for EMA calculations
    # @param[out]         out: MACD component DataFrame
    #
    # @callsequence       @startuml
    #                     title TrendStrength.compute_macd_components
    #                     start
    #                     :Validate all MACD windows;
    #                     if (window_fast >= window_slow?) then (yes)
    #                       :Raise ValueError;
    #                       stop
    #                     endif
    #                     :Convert input to pandas Series;
    #                     :Compute fast and slow EMA;
    #                     :Compute MACD line, signal line, and histogram;
    #                     :Return MACD DataFrame;
    #                     stop
    #                     @enduml
    # ***********************************************************************************************************************
    @staticmethod
    def compute_macd_components(
        series,
        window_slow: int = 26,
        window_fast: int = 12,
        window_signal: int = 9,
        min_periods: int | None = None,
    ) -> pd.DataFrame:
        validate_window(window_slow, "window_slow")
        validate_window(window_fast, "window_fast")
        validate_window(window_signal, "window_signal")

        if window_fast >= window_slow:
            raise ValueError("window_fast should be smaller than window_slow.")

        s = to_series(series)
        mp = None if min_periods is None else min_periods

        # MACD line = difference between fast and slow EMAs.
        ema_fast = ema_series(s, window_fast, adjust=False, min_periods=mp)
        ema_slow = ema_series(s, window_slow, adjust=False, min_periods=mp)

        macd_line = ema_fast - ema_slow
        signal_line = ema_series(macd_line, window_signal, adjust=False, min_periods=mp)
        hist = macd_line - signal_line
        return pd.DataFrame({"macd": macd_line, "macd_signal": signal_line, "macd_hist": hist}, index=s.index)

    # ***********************************************************************************************************************
    # Functionname:       TrendStrength.compute_macd(series, window_slow: int = 26, window_fast: int = 12,
    #                              window_signal: int = 9, min_periods: int | None = None)
    #
    # @brief              Return MACD line only.
    # @pre                Same preconditions as compute_macd_components.
    # @post               Returns macd series.
    # @param[in]          series: Input signal
    #                     window_slow: Slow EMA window
    #                     window_fast: Fast EMA window
    #                     window_signal: Signal EMA window
    #                     min_periods: Minimum periods for EMA calculations
    # @param[out]         out: MACD line series
    #
    # @callsequence       @startuml
    #                     title TrendStrength.compute_macd
    #                     start
    #                     :Call compute_macd_components;
    #                     :Extract macd column;
    #                     :Return MACD series;
    #                     stop
    #                     @enduml
    # ***********************************************************************************************************************
    @staticmethod
    def compute_macd(
        series,
        window_slow: int = 26,
        window_fast: int = 12,
        window_signal: int = 9,
        min_periods: int | None = None,
    ) -> pd.Series:
        return TrendStrength.compute_macd_components(
            series,
            window_slow=window_slow,
            window_fast=window_fast,
            window_signal=window_signal,
            min_periods=min_periods,
        )["macd"]

    # ***********************************************************************************************************************
    # Functionname:       TrendStrength.compute_macd_signal(series, window_slow: int = 26,
    #                              window_fast: int = 12, window_signal: int = 9,
    #                              min_periods: int | None = None)
    #
    # @brief              Return MACD signal line only.
    # @pre                Same preconditions as compute_macd_components.
    # @post               Returns macd_signal series.
    # @param[in]          series: Input signal
    #                     window_slow: Slow EMA window
    #                     window_fast: Fast EMA window
    #                     window_signal: Signal EMA window
    #                     min_periods: Minimum periods for EMA calculations
    # @param[out]         out: MACD signal series
    #
    # @callsequence       @startuml
    #                     title TrendStrength.compute_macd_signal
    #                     start
    #                     :Call compute_macd_components;
    #                     :Extract macd_signal column;
    #                     :Return MACD signal series;
    #                     stop
    #                     @enduml
    # ***********************************************************************************************************************
    @staticmethod
    def compute_macd_signal(
        series,
        window_slow: int = 26,
        window_fast: int = 12,
        window_signal: int = 9,
        min_periods: int | None = None,
    ) -> pd.Series:
        return TrendStrength.compute_macd_components(
            series,
            window_slow=window_slow,
            window_fast=window_fast,
            window_signal=window_signal,
            min_periods=min_periods,
        )["macd_signal"]

    # ***********************************************************************************************************************
    # Functionname:       TrendStrength.compute_macd_hist(series, window_slow: int = 26,
    #                              window_fast: int = 12, window_signal: int = 9,
    #                              min_periods: int | None = None)
    #
    # @brief              Return MACD histogram only.
    # @pre                Same preconditions as compute_macd_components.
    # @post               Returns macd_hist series.
    # @param[in]          series: Input signal
    #                     window_slow: Slow EMA window
    #                     window_fast: Fast EMA window
    #                     window_signal: Signal EMA window
    #                     min_periods: Minimum periods for EMA calculations
    # @param[out]         out: MACD histogram series
    #
    # @callsequence       @startuml
    #                     title TrendStrength.compute_macd_hist
    #                     start
    #                     :Call compute_macd_components;
    #                     :Extract macd_hist column;
    #                     :Return MACD histogram series;
    #                     stop
    #                     @enduml
    # ***********************************************************************************************************************
    @staticmethod
    def compute_macd_hist(
        series,
        window_slow: int = 26,
        window_fast: int = 12,
        window_signal: int = 9,
        min_periods: int | None = None,
    ) -> pd.Series:
        return TrendStrength.compute_macd_components(
            series,
            window_slow=window_slow,
            window_fast=window_fast,
            window_signal=window_signal,
            min_periods=min_periods,
        )["macd_hist"]

    # ***********************************************************************************************************************
    # Functionname:       TrendStrength.compute_plus_di(high, low, close, window: int = 14)
    #
    # @brief              Return positive directional indicator (+DI).
    # @pre                window > 0 and aligned high/low/close inputs
    # @post               Returns plus_di series.
    # @param[in]          high: High price series
    #                     low: Low price series
    #                     close: Close price series
    #                     window: DI/ADX window
    # @param[out]         out: +DI series
    #
    # @callsequence       @startuml
    #                     title TrendStrength.compute_plus_di
    #                     start
    #                     :Call compute_directional_indicators;
    #                     :Extract plus_di column;
    #                     :Return +DI series;
    #                     stop
    #                     @enduml
    # ***********************************************************************************************************************
    @staticmethod
    def compute_plus_di(high, low, close, window: int = 14) -> pd.Series:
        return TrendStrength.compute_directional_indicators(high, low, close, window=window)["plus_di"]

    # ***********************************************************************************************************************
    # Functionname:       TrendStrength.compute_minus_di(high, low, close, window: int = 14)
    #
    # @brief              Return negative directional indicator (-DI).
    # @pre                window > 0 and aligned high/low/close inputs
    # @post               Returns minus_di series.
    # @param[in]          high: High price series
    #                     low: Low price series
    #                     close: Close price series
    #                     window: DI/ADX window
    # @param[out]         out: -DI series
    #
    # @callsequence       @startuml
    #                     title TrendStrength.compute_minus_di
    #                     start
    #                     :Call compute_directional_indicators;
    #                     :Extract minus_di column;
    #                     :Return -DI series;
    #                     stop
    #                     @enduml
    # ***********************************************************************************************************************
    @staticmethod
    def compute_minus_di(high, low, close, window: int = 14) -> pd.Series:
        return TrendStrength.compute_directional_indicators(high, low, close, window=window)["minus_di"]

    # ***********************************************************************************************************************
    # Functionname:       TrendStrength.compute_adx(high, low, close, window: int = 14)
    #
    # @brief              Return Average Directional Index (ADX).
    # @pre                window > 0 and aligned high/low/close inputs
    # @post               Returns adx series.
    # @param[in]          high: High price series
    #                     low: Low price series
    #                     close: Close price series
    #                     window: DI/ADX window
    # @param[out]         out: ADX series
    #
    # @callsequence       @startuml
    #                     title TrendStrength.compute_adx
    #                     start
    #                     :Call compute_directional_indicators;
    #                     :Extract adx column;
    #                     :Return ADX series;
    #                     stop
    #                     @enduml
    # ***********************************************************************************************************************
    @staticmethod
    def compute_adx(high, low, close, window: int = 14) -> pd.Series:
        return TrendStrength.compute_directional_indicators(high, low, close, window=window)["adx"]

    # ***********************************************************************************************************************
    # Functionname:       TrendStrength.compute_directional_indicators(high, low, close, window: int = 14)
    #
    # @brief              Compute +DI, -DI, DX, and ADX.
    # @pre                window > 0 and aligned high/low/close inputs
    # @post               Returns DataFrame with plus_di, minus_di, dx, and adx.
    # @param[in]          high: High price series
    #                     low: Low price series
    #                     close: Close price series
    #                     window: DI/ADX window
    # @param[out]         out: Directional indicator DataFrame
    #
    # @callsequence       @startuml
    #                     title TrendStrength.compute_directional_indicators
    #                     start
    #                     :Validate window;
    #                     :Convert high, low, and close to pandas Series;
    #                     :Compute directional movements and true range;
    #                     :Apply Wilder smoothing to TR and directional movement;
    #                     :Compute plus_di and minus_di;
    #                     :Compute dx and adx;
    #                     :Return directional indicators DataFrame;
    #                     stop
    #                     @enduml
    # ***********************************************************************************************************************
    @staticmethod
    def compute_directional_indicators(high, low, close, window: int = 14) -> pd.DataFrame:
        validate_window(window, "window")

        h = to_series(high, name="high")
        l = to_series(low, name="low")
        c = to_series(close, name="close")

        up_move = h.diff()
        down_move = -l.diff()

        # Directional movement: only take positive leg that exceeds the opposite.
        plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=h.index, name="plus_dm")
        minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=h.index, name="minus_dm")

        tr = true_range(h, l, c)

        atr = tr.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
        plus_dm_smooth = plus_dm.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
        minus_dm_smooth = minus_dm.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()

        plus_di = 100.0 * (plus_dm_smooth / atr)
        minus_di = 100.0 * (minus_dm_smooth / atr)

        # ADX is the EMA of the normalized DI spread.
        di_sum = plus_di + minus_di
        dx = 100.0 * ((plus_di - minus_di).abs() / di_sum)
        dx = dx.replace([np.inf, -np.inf], np.nan)
        adx = dx.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()

        return pd.DataFrame({"plus_di": plus_di,
                             "minus_di": minus_di,
                             "dx": dx,
                             "adx": adx}, index=h.index)

    # ***********************************************************************************************************************
    # Functionname:       TrendStrength.compute_trend_helper_signal(high, low, close,
    #                              adx_window: int = 14, adx_threshold: float = 20.0,
    #                              macd_window_slow: int = 26, macd_window_fast: int = 12,
    #                              macd_window_signal: int = 9,
    #                              min_periods: int | None = None)
    #
    # @brief              Build helper-oriented trend regime signal from MACD and ADX/DI confirmation.
    # @pre                Valid windows, adx_threshold >= 0, and aligned OHLC inputs.
    # @post               Returns DataFrame with trend components and position in {-1.0, 0.0, 1.0}.
    # @param[in]          high: High price series
    #                     low: Low price series
    #                     close: Close price series
    #                     adx_window: DI/ADX smoothing window
    #                     adx_threshold: Minimum ADX required to emit directional position
    #                     macd_window_slow: MACD slow EMA window
    #                     macd_window_fast: MACD fast EMA window
    #                     macd_window_signal: MACD signal EMA window
    #                     min_periods: Minimum periods for MACD EMA calculations
    # @param[out]         out: DataFrame with macd, macd_signal, macd_hist, plus_di, minus_di, adx, position
    #
    # @callsequence       @startuml
    #                     title TrendStrength.compute_trend_helper_signal
    #                     start
    #                     :Validate adx_threshold >= 0;
    #                     :Convert close to pandas Series;
    #                     :Compute MACD components;
    #                     :Compute directional indicators (DI/ADX);
    #                     :Concatenate MACD and DI columns;
    #                     :Apply long/short position conditions;
    #                     :Return DataFrame with indicators and position;
    #                     stop
    #                     @enduml
    # ***********************************************************************************************************************
    @staticmethod
    def compute_trend_helper_signal(high,
        low,
        close,
        adx_window: int = 14,
        adx_threshold: float = 20.0,
        macd_window_slow: int = 26,
        macd_window_fast: int = 12,
        macd_window_signal: int = 9,
        min_periods: int | None = None) -> pd.DataFrame:
        if adx_threshold < 0:
            raise ValueError("adx_threshold must be >= 0")

        c       = to_series(close, name="close")
        macd_df = TrendStrength.compute_macd_components(c, 
                                                        window_slow=macd_window_slow, 
                                                        window_fast=macd_window_fast, 
                                                        window_signal=macd_window_signal,
                                                        min_periods=min_periods)
        di_df = TrendStrength.compute_directional_indicators(high, low, c, window=adx_window)
        out   = pd.concat([macd_df[["macd", "macd_signal", "macd_hist"]], 
                           di_df[["plus_di", "minus_di", "adx"]]],
                           axis=1)
        long_cond  = (out["macd"] > out["macd_signal"]) & (out["plus_di"] > out["minus_di"]) & (out["adx"] >= adx_threshold)
        short_cond = (out["macd"] < out["macd_signal"]) & (out["minus_di"] > out["plus_di"]) & (out["adx"] >= adx_threshold)
        position   = pd.Series(0.0, index=out.index, name="position")
        position[long_cond] = 1.0
        position[short_cond] = -1.0

        return pd.concat([out, position], axis=1)

    # ***********************************************************************************************************************
    # Functionname:       TrendStrength.classify_trend_block(block_df: pd.DataFrame,
    #                              open_col: str = "open", close_col: str = "close")
    #
    # @brief              Classify a price block as bullish, bearish, or neutral.
    # @pre                block_df must contain open_col and close_col with at least one row.
    # @post               Returns one label in {"alcista", "bajista", "lateral"}.
    # @param[in]          block_df: Block-level OHLC data
    #                     open_col: Open column name
    #                     close_col: Close column name
    # @param[out]         out: Trend label string
    #
    # @callsequence       @startuml
    #                     title TrendStrength.classify_trend_block
    #                     start
    #                     if (block_df empty or missing columns?) then (yes)
    #                       :Raise ValueError;
    #                       stop
    #                     endif
    #                     :Extract first open and last close;
    #                     if (close > open?) then (yes)
    #                       :Return "alcista";
    #                     elseif (close < open?) then (yes)
    #                       :Return "bajista";
    #                     else
    #                       :Return "lateral";
    #                     endif
    #                     stop
    #                     @enduml
    # ***********************************************************************************************************************
    @staticmethod
    def classify_trend_block(block_df: pd.DataFrame, open_col: str = "open", close_col: str = "close") -> str:
        if block_df.empty:
            raise ValueError("block_df must not be empty")
        if open_col not in block_df.columns or close_col not in block_df.columns:
            raise ValueError(f"block_df must contain '{open_col}' and '{close_col}' columns")
        open_macro  = float(block_df[open_col].iloc[0])
        close_macro = float(block_df[close_col].iloc[-1])
        if close_macro > open_macro:
            return "alcista"
        if close_macro < open_macro:
            return "bajista"
        return "lateral"

    # ***********************************************************************************************************************
    # Functionname:       TrendStrength.compute_weighted_trend_score(new_trend: str, history: list[str])
    #
    # @brief              Compute continuous trend score from categorical history using decreasing weights.
    # @pre                new_trend and history labels must be in {"alcista", "lateral", "bajista"}.
    # @post               Appends new_trend into history and returns score in [-1.0, 1.0].
    # @param[in]          new_trend: Current trend label
    #                     history: Mutable trend history list
    # @param[out]         out: Weighted trend score
    #
    # @callsequence       @startuml
    #                       start
    #                        if (new_trend not in valid labels?) then (yes)
    #                           :Raise ValueError;
    #                           stop
    #                       endif
    #                       :Append new_trend to history;
    #                       :Select weight vector based on history length;
    #                       :Normalize weights to sum=1;
    #                       :Map history labels to numeric scores;
    #                       :Return weighted dot product score;
    #                       stop
    #                     @enduml
    # ***********************************************************************************************************************
    @staticmethod
    def compute_weighted_trend_score(new_trend: str, history: list[str]) -> float:
        mapping = {"alcista": 1.0, "lateral": 0.0, "bajista": -1.0}
        if new_trend not in mapping:
            raise ValueError("new_trend must be one of {'alcista', 'lateral', 'bajista'}")

        history.append(new_trend)
        n = len(history)

        if n == 1:
            weights = [1.0]
        elif n == 2:
            weights = [0.75, 0.25]
        elif n == 3:
            weights = [0.66, 0.22, 0.12]
        elif n == 4:
            weights = [0.6, 0.2, 0.13, 0.07]
        elif n == 5:
            weights = [0.55, 0.15, 0.12, 0.1, 0.08]
        else:
            base = 0.5 / (n - 1)
            weights = [0.5] + [base] * (n - 1)

        weights = np.asarray(weights, dtype=float)
        weights = weights / np.sum(weights)

        try:
            scores = np.asarray([mapping[item] for item in history], dtype=float)
        except KeyError as exc:
            raise ValueError("history labels must be one of {'alcista', 'lateral', 'bajista'}") from exc

        return float(np.dot(weights, scores))

    # ***********************************************************************************************************************
    # Functionname:       TrendStrength.build_trending_struct(data: pd.DataFrame, fulltrend_col: str = "FullTrend",
    #                              q_strong: float = 0.90, q_mid: float = 0.65)
    #
    # @brief              Build threshold structure and current label from a FullTrend-like score series.
    # @pre                fulltrend_col exists, has non-NaN values, and 0 <= q_mid <= q_strong <= 1.
    # @post               Returns dict with levels, last_value, and signal.
    # @param[in]          data: Source DataFrame
    #                     fulltrend_col: Column containing trend score
    #                     q_strong: High quantile for strong thresholds
    #                     q_mid: Mid quantile for regular thresholds
    # @param[out]         out: Structure with levels and current categorical signal
    #
    # @callsequence       @startuml
    #                     title TrendStrength.build_trending_struct
    #                     start
    #                     if (column missing or invalid quantiles?) then (yes)
    #                       :Raise ValueError;
    #                       stop
    #                     endif
    #                     :Compute strong and mid quantile thresholds;
    #                     :Build levels dict with symmetric buy/sell levels;
    #                     :Classify last value against levels;
    #                     :Return structure with levels and signal;
    #                     stop
    #                     @enduml
    # ***********************************************************************************************************************
    @staticmethod
    def build_trending_struct(
        data: pd.DataFrame,
        fulltrend_col: str = "FullTrend",
        q_strong: float = 0.90,
        q_mid: float = 0.65,
    ) -> dict[str, object]:
        if fulltrend_col not in data.columns:
            raise ValueError(f"column '{fulltrend_col}' not found in DataFrame")
        if not (0.0 <= q_mid <= q_strong <= 1.0):
            raise ValueError("Require 0 <= q_mid <= q_strong <= 1")

        ft = data[fulltrend_col].dropna()
        if ft.empty:
            raise ValueError(f"column '{fulltrend_col}' has no valid values")

        strong = float(ft.quantile(q_strong))
        mid = float(ft.quantile(q_mid))

        levels = {
            "strong_buy": strong,
            "buy": mid,
            "sell": -mid,
            "strong_sell": -strong,
        }

        last = float(ft.iloc[-1])
        if last >= levels["strong_buy"]:
            signal = "STRONG BUY"
        elif last >= levels["buy"]:
            signal = "BUY"
        elif last <= levels["strong_sell"]:
            signal = "STRONG SELL"
        elif last <= levels["sell"]:
            signal = "SELL"
        else:
            signal = "NEUTRAL"

        return {"levels": levels, "last_value": last, "signal": signal}

    # ***********************************************************************************************************************
    # Functionname:       TrendStrength.compute_block_trend_profile(data: pd.DataFrame, step: int = 5,
    #                              sensitivity: float = 10.0, open_col: str = "open", close_col: str = "close")
    #
    # @brief              Compute piecewise-constant trend profile over non-overlapping blocks.
    # @pre                step > 0, sensitivity > 0, and input columns exist.
    # @post               Returns Series aligned to input index with values in [-1, 1].
    # @param[in]          data: Source OHLC DataFrame
    #                     step: Block size
    #                     sensitivity: tanh scaling for block return
    #                     open_col: Open column name
    #                     close_col: Close column name
    # @param[out]         out: Block trend score series
    #
    # @callsequence       @startuml
    #                     title TrendStrength.compute_block_trend_profile
    #                     start
    #                     :Validate step and sensitivity;
    #                     if (data empty?) then (yes)
    #                       :Return empty series;
    #                       stop
    #                     endif
    #                     repeat
    #                       :Extract non-overlapping block;
    #                       :Compute return and tanh-scaled score;
    #                       :Replicate score across block indices;
    #                     repeat while (more complete blocks?)
    #                     :Pad remaining positions with last score;
    #                     :Return block trend score series;
    #                     stop
    #                     @enduml
    # ***********************************************************************************************************************
    @staticmethod
    def compute_block_trend_profile(
        data: pd.DataFrame,
        step: int = 5,
        sensitivity: float = 10.0,
        open_col: str = "open",
        close_col: str = "close",
    ) -> pd.Series:
        validate_window(step, "step")
        if sensitivity <= 0:
            raise ValueError("sensitivity must be > 0")
        if open_col not in data.columns or close_col not in data.columns:
            raise ValueError(f"data must contain '{open_col}' and '{close_col}' columns")

        if data.empty:
            return pd.Series(dtype=float, index=data.index, name=f"trend_profile_{step}")

        scores: list[float] = []
        n = len(data)

        for i in range(0, n, step):
            block = data.iloc[i : i + step]
            if len(block) < step:
                break
            open_val = float(block[open_col].iloc[0])
            close_val = float(block[close_col].iloc[-1])
            ret = (close_val - open_val) / open_val
            block_score = float(np.tanh(ret * sensitivity))
            scores.extend([block_score] * step)

        if len(scores) == 0:
            scores = [0.0] * n
        elif len(scores) < n:
            scores.extend([scores[-1]] * (n - len(scores)))

        return pd.Series(scores[:n], index=data.index, name=f"trend_profile_{step}")


# ---------------------------------------------------------------------------
# Backward-compatible module-level wrappers
# ---------------------------------------------------------------------------
def compute_macd_components(series, window_slow: int = 26, window_fast: int = 12, window_signal: int = 9, min_periods: int | None = None) -> pd.DataFrame:
    return TrendStrength.compute_macd_components(series, window_slow=window_slow, window_fast=window_fast, window_signal=window_signal, min_periods=min_periods)


def compute_macd(series, window_slow: int = 26, window_fast: int = 12, window_signal: int = 9, min_periods: int | None = None) -> pd.Series:
    return TrendStrength.compute_macd(series, window_slow=window_slow, window_fast=window_fast, window_signal=window_signal, min_periods=min_periods)


def compute_macd_signal(series, window_slow: int = 26, window_fast: int = 12, window_signal: int = 9, min_periods: int | None = None) -> pd.Series:
    return TrendStrength.compute_macd_signal(series, window_slow=window_slow, window_fast=window_fast, window_signal=window_signal, min_periods=min_periods)


def compute_macd_hist(series, window_slow: int = 26, window_fast: int = 12, window_signal: int = 9, min_periods: int | None = None) -> pd.Series:
    return TrendStrength.compute_macd_hist(series, window_slow=window_slow, window_fast=window_fast, window_signal=window_signal, min_periods=min_periods)


def compute_plus_di(high, low, close, window: int = 14) -> pd.Series:
    return TrendStrength.compute_plus_di(high, low, close, window=window)


def compute_minus_di(high, low, close, window: int = 14) -> pd.Series:
    return TrendStrength.compute_minus_di(high, low, close, window=window)


def compute_adx(high, low, close, window: int = 14) -> pd.Series:
    return TrendStrength.compute_adx(high, low, close, window=window)


def compute_directional_indicators(high, low, close, window: int = 14) -> pd.DataFrame:
    return TrendStrength.compute_directional_indicators(high, low, close, window=window)


def compute_trend_helper_signal(
    high,
    low,
    close,
    adx_window: int = 14,
    adx_threshold: float = 20.0,
    macd_window_slow: int = 26,
    macd_window_fast: int = 12,
    macd_window_signal: int = 9,
    min_periods: int | None = None,
) -> pd.DataFrame:
    return TrendStrength.compute_trend_helper_signal(
        high=high,
        low=low,
        close=close,
        adx_window=adx_window,
        adx_threshold=adx_threshold,
        macd_window_slow=macd_window_slow,
        macd_window_fast=macd_window_fast,
        macd_window_signal=macd_window_signal,
        min_periods=min_periods,
    )


def classify_trend_block(block_df: pd.DataFrame, open_col: str = "open", close_col: str = "close") -> str:
    return TrendStrength.classify_trend_block(block_df=block_df, open_col=open_col, close_col=close_col)


def compute_weighted_trend_score(new_trend: str, history: list[str]) -> float:
    return TrendStrength.compute_weighted_trend_score(new_trend=new_trend, history=history)


def build_trending_struct(
    data: pd.DataFrame,
    fulltrend_col: str = "FullTrend",
    q_strong: float = 0.90,
    q_mid: float = 0.65,
) -> dict[str, object]:
    return TrendStrength.build_trending_struct(
        data=data,
        fulltrend_col=fulltrend_col,
        q_strong=q_strong,
        q_mid=q_mid,
    )


def compute_block_trend_profile(
    data: pd.DataFrame,
    step: int = 5,
    sensitivity: float = 10.0,
    open_col: str = "open",
    close_col: str = "close",
) -> pd.Series:
    return TrendStrength.compute_block_trend_profile(
        data=data,
        step=step,
        sensitivity=sensitivity,
        open_col=open_col,
        close_col=close_col,
    )


__all__ = [
    "TrendStrength",
    "compute_macd_components",
    "compute_macd",
    "compute_macd_signal",
    "compute_macd_hist",
    "compute_plus_di",
    "compute_minus_di",
    "compute_adx",
    "compute_directional_indicators",
    "compute_trend_helper_signal",
    "classify_trend_block",
    "compute_weighted_trend_score",
    "build_trending_struct",
    "compute_block_trend_profile",
]
