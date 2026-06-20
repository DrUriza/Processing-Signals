from __future__   import annotations
import numpy      as np
import pandas     as pd
from scipy.signal import argrelextrema

# \file **********************************************************************
# COMPANY:            ELATIN
# PROJECT:            SIGNAL_ANALYSIS
# COMPONENT:          ADAPTERS
# MODULE NAME:        patterns.py
# DESCRIPTION:        @brief Chart/candlestick pattern utilities for OHLC data
# CREATION DATE:      22.04.2026
# VERSION:            $Revision: 0.1$
# CHANGES:            22.04.2026 - Added H&S, inverse H&S, pennants, and candle patterns.
# *****************************************************************************

class PatternAdapters:
    # ***********************************************************************************************************************
    # Functionname:       PatternAdapters.compute_candlestick_patterns(ohlc: pd.DataFrame,
    #                              doji_threshold: float = 0.1, shadow_ratio: float = 2.0)
    #
    # @brief              Compute candlestick pattern flags and coarse candle class labels.
    # @pre                ohlc contains open, high, low, close columns
    # @post               Returns DataFrame with boolean flags and candle_class
    # @param[in]          ohlc: OHLC DataFrame
    # @param[out]         result: Candlestick feature DataFrame
    #
    # @callsequence       @startuml
    #                     title PatternAdapters.compute_candlestick_patterns
    #                     start
    #                     :Call _validate_ohlc_dataframe(ohlc);
    #                     :Compute body, total_range,
    #                     upper_shadow, lower_shadow;
    #                     :Compute body/upper/lower ratios;
    #                     :Build bullish/bearish condition masks;
    #                     :Map masks to candle_class labels;
    #                     :Compute doji and hammer flags;
    #                     :Compute bullish/bearish engulfing flags
    #                     with shifted open/close;
    #                     :Return DataFrame with candle_class,
    #                     doji, hammer, bullish_engulfing,
    #                     bearish_engulfing;
    #                     end
    #                     @enduml
    #
    # @InOutCorrelation   As described in UML diagram
    # @traceability
    # ***********************************************************************************************************************
    @staticmethod
    def compute_candlestick_patterns(ohlc: pd.DataFrame,
                                     doji_threshold: float = 0.1,
                                     shadow_ratio: float = 2.0) -> pd.DataFrame:
        PatternAdapters._validate_ohlc_dataframe(ohlc)

        body         = (ohlc["close"] - ohlc["open"]).abs()
        total_range  = (ohlc["high"] - ohlc["low"]).replace(0.0, np.nan)
        upper_shadow = ohlc["high"] - ohlc[["open", "close"]].max(axis=1)
        lower_shadow = ohlc[["open", "close"]].min(axis=1) - ohlc["low"]

        body_ratio   = body / total_range
        upper_ratio  = upper_shadow / total_range
        lower_ratio  = lower_shadow / total_range
        doji         = body_ratio <= doji_threshold

        bullish = ohlc["close"] > ohlc["open"]
        bearish = ~bullish
        conditions = [
            bullish & (body_ratio > 0.7) & (upper_ratio < 0.1) & (lower_ratio > 0.2),
            bullish & (body_ratio > 0.6) & (lower_ratio > upper_ratio * 1.5),
            bullish & (body_ratio > 0.6) & (upper_ratio > lower_ratio * 1.5),
            bullish & (body_ratio > 0.4),
            bullish,
            bearish & (body_ratio > 0.7) & (lower_ratio < 0.1) & (upper_ratio > 0.2),
            bearish & (body_ratio > 0.6) & (upper_ratio > lower_ratio * 1.5),
            bearish & (body_ratio > 0.6) & (lower_ratio > upper_ratio * 1.5),
            bearish & (body_ratio > 0.4),
            bearish,
        ]
        choices = [
            "Strong Bullish",
            "Strong Bullish Lower",
            "Bullish Resistance",
            "Bullish",
            "Low Bullish",
            "Strong Bearish",
            "Strong Bearish Upper",
            "Bearish Support",
            "Bearish",
            "Low Bearish",
        ]
        candle_class = pd.Series(np.select(conditions, choices, default="Unknown"), index=ohlc.index)

        # Hammer: small body near top with long lower shadow.
        hammer = (lower_shadow >= shadow_ratio * body.replace(0.0, np.nan)) & (upper_shadow <= body)
        hammer = hammer.fillna(False)

        prev_open = ohlc["open"].shift(1)
        prev_close = ohlc["close"].shift(1)

        prev_bearish = prev_close < prev_open
        prev_bullish = prev_close > prev_open
        curr_bullish = ohlc["close"] > ohlc["open"]
        curr_bearish = ohlc["close"] < ohlc["open"]

        bullish_engulfing = (prev_bearish
            & curr_bullish
            & (ohlc["open"] <= prev_close)
            & (ohlc["close"] >= prev_open))
        bearish_engulfing = (prev_bullish
            & curr_bearish
            & (ohlc["open"] >= prev_close)
            & (ohlc["close"] <= prev_open)
        )

        return pd.DataFrame(
            {
                "candle_class": candle_class,
                "doji": doji.fillna(False),
                "hammer": hammer,
                "bullish_engulfing": bullish_engulfing.fillna(False),
                "bearish_engulfing": bearish_engulfing.fillna(False),
            },
            index=ohlc.index,
        )

    # ***********************************************************************************************************************
    # Functionname:       PatternAdapters.build_head_shoulders_struct(ohlc: pd.DataFrame,
    #                              order: int = 10, rr: float = 1.8)
    #
    # @brief              Detect H&S and inverse H&S and return chart-ready structure.
    # @pre                ohlc contains close and rr > 0 and order > 0
    # @post               Returns dict with HCH and HCHi arrays
    # @param[in]          ohlc: OHLC DataFrame
    # @param[out]         result: Structured H&S pattern dict
    #
    # @callsequence       @startuml
    #                     title PatternAdapters.build_head_shoulders_struct
    #                     start
    #                     :Call _validate_ohlc_dataframe(ohlc);
    #                     :Validate order > 0 and rr > 0;
    #                     :Extract close prices;
    #                     :Find local highs and lows with argrelextrema;
    #                     :Initialize empty signals list;
    #                     repeat
    #                       :Scan triplets of highs;
    #                       if (middle high is head?) then (yes)
    #                         :Estimate neckline from interim lows;
    #                         :Append H&S signal;
    #                       endif
    #                     repeat while (high triplets remaining)
    #                     repeat
    #                       :Scan triplets of lows;
    #                       if (middle low is inverse head?) then (yes)
    #                         :Estimate neckline from interim highs;
    #                         :Append H&S_INV signal;
    #                       endif
    #                     repeat while (low triplets remaining)
    #                     :Initialize patterns dict with HCH/HCHi;
    #                     repeat
    #                       :Convert each signal to chart structure
    #                       with entry, stop, and exit;
    #                     repeat while (signals remaining)
    #                     :Return patterns;
    #                     end
    #                     @enduml
    #
    # @InOutCorrelation   As described in UML diagram
    # @traceability
    # ***********************************************************************************************************************
    @staticmethod
    def build_head_shoulders_struct(ohlc: pd.DataFrame,
                                    order: int = 10,
                                    rr: float = 1.8) -> dict[str, list[dict]]:
        PatternAdapters._validate_ohlc_dataframe(ohlc)
        if not isinstance(order, int) or order <= 0:
            raise ValueError("order must be a positive integer.")
        if rr <= 0:
            raise ValueError("rr must be > 0")

        prices = ohlc["close"].to_numpy(dtype=float)
        highs = argrelextrema(prices, np.greater, order=order)[0]
        lows = argrelextrema(prices, np.less, order=order)[0]

        signals: list[dict] = []
        for i in range(len(highs) - 2):
            h1, h2, h3 = highs[i:i + 3]
            if prices[h2] > prices[h1] and prices[h2] > prices[h3]:
                mid_lows = [prices[l] for l in lows if h1 < l < h3]
                if len(mid_lows) == 0:
                    continue
                signals.append({"type": "H&S", "pattern": (int(h1), int(h2), int(h3), float(np.mean(mid_lows)))})

        for i in range(len(lows) - 2):
            l1, l2, l3 = lows[i:i + 3]
            if prices[l2] < prices[l1] and prices[l2] < prices[l3]:
                mid_highs = [prices[h] for h in highs if l1 < h < l3]
                if len(mid_highs) == 0:
                    continue
                signals.append({"type": "H&S_INV", "pattern": (int(l1), int(l2), int(l3), float(np.mean(mid_highs)))})

        patterns = {"HCH": [], "HCHi": []}
        idx = list(ohlc.index)

        for signal in signals:
            p1_i, p2_i, p3_i, neckline = signal["pattern"]
            p1, p2, p3 = prices[[p1_i, p2_i, p3_i]]

            if signal["type"] == "H&S":
                entry_price = neckline * 0.997
                stop_loss = p2 * 1.01
                exit_price = entry_price - (stop_loss - entry_price) * rr
                patterns["HCH"].append({
                    "x": [idx[p1_i], idx[p2_i], idx[p3_i]],
                    "y": [float(p1), float(p2), float(p3)],
                    "neckline": [float(neckline)] * 2,
                    "neck_x": [idx[p1_i], idx[p3_i]],
                    "entry_x": idx[p3_i],
                    "entry_y": float(entry_price),
                    "stop_y": float(stop_loss),
                    "exit_y": float(exit_price),
                    "type": "bearish"})

            if signal["type"] == "H&S_INV":
                entry_price = neckline * 1.003
                stop_loss = p2 * 0.99
                exit_price = entry_price + (entry_price - stop_loss) * rr
                patterns["HCHi"].append({
                    "x": [idx[p1_i], idx[p2_i], idx[p3_i]],
                    "y": [float(p1), float(p2), float(p3)],
                    "neckline": [float(neckline)] * 2,
                    "neck_x": [idx[p1_i], idx[p3_i]],
                    "entry_x": idx[p3_i],
                    "entry_y": float(entry_price),
                    "stop_y": float(stop_loss),
                    "exit_y": float(exit_price),
                    "type": "bullish"})
        return patterns

    # ***********************************************************************************************************************
    # Functionname:       PatternAdapters.build_pennants_struct(ohlc: pd.DataFrame, lookback: int = 60,
    #                              min_touch: int = 3, max_squeeze: float = 0.35, rng: int = 40)
    #
    # @brief              Detect pennant breakouts and return structured chart objects.
    # @pre                ohlc contains high, low, close columns
    # @post               Returns list of pennant objects
    # @param[in]          ohlc: OHLC DataFrame
    # @param[out]         result: Pennant object list
    #
    # @callsequence       @startuml
    #                     title PatternAdapters.build_pennants_struct
    #                     start
    #                     :Call _validate_ohlc_dataframe(ohlc);
    #                     :Validate lookback/min_touch/rng constraints;
    #                     :Extract close/high/low arrays;
    #                     :Initialize empty pennants list;
    #                     repeat
    #                       :Select rolling segment of size lookback;
    #                       :Fit upper/lower trendlines (polyfit);
    #                       if (not converging?) then (yes)
    #                         :Skip segment;
    #                       else (no)
    #                         :Compute spread and squeeze;
    #                         :Check squeeze and touch-count thresholds;
    #                         :Check breakout direction (bull/bear);
    #                         if (breakout valid?) then (yes)
    #                           :Build pennant dict and append;
    #                         endif
    #                       endif
    #                     repeat while (bars remaining)
    #                     :Return pennants list;
    #                     end
    #                     @enduml
    #
    # @InOutCorrelation   As described in UML diagram
    # @traceability
    # ***********************************************************************************************************************
    @staticmethod
    def build_pennants_struct(ohlc: pd.DataFrame, lookback: int = 60, min_touch: int = 3, max_squeeze: float = 0.35, rng: int = 40) -> list[dict]:
        PatternAdapters._validate_ohlc_dataframe(ohlc)

        if lookback <= 1 or min_touch <= 0 or rng <= 0:
            raise ValueError("lookback/min_touch/rng must be positive and lookback > 1.")

        closes = ohlc["close"].to_numpy(dtype=float)
        highs  = ohlc["high"].to_numpy(dtype=float)
        lows   = ohlc["low"].to_numpy(dtype=float)
        idx    = list(ohlc.index)
        pennants: list[dict] = []
        for i in range(lookback, len(ohlc)):
            seg_high = highs[i - lookback : i]
            seg_low  = lows[i - lookback : i]
            x        = np.arange(lookback)
            up_slope, up_int = np.polyfit(x, seg_high, 1)
            dn_slope, dn_int = np.polyfit(x, seg_low, 1)
            # For a pennant we need converging lines.
            if up_slope >= 0 or dn_slope <= 0:
                continue
            spread_start = seg_high[0] - seg_low[0]
            spread_end = seg_high[-1] - seg_low[-1]
            if spread_start <= 0:
                continue
            squeeze = spread_end / spread_start
            if squeeze > max_squeeze:
                continue
            touches_up = int(np.sum(seg_high >= (up_slope * x + up_int) * 0.995))
            touches_dn = int(np.sum(seg_low <= (dn_slope * x + dn_int) * 1.005))
            if touches_up < min_touch or touches_dn < min_touch:
                continue
            last_close = closes[i]
            up_line    = up_slope * lookback + up_int
            dn_line    = dn_slope * lookback + dn_int
            if last_close > up_line:
                direction = "bull"
            elif last_close < dn_line:
                direction = "bear"
            else:
                continue
            a = max(0, i - rng)
            b = i
            c = min(len(ohlc) - 1, i + rng)
            pennants.append({
                    "index": int(i),
                    "type": direction,
                    "upper_x": [idx[a], idx[b], idx[c]],
                    "upper_y": [float(closes[a] + up_slope * rng),
                                float(closes[b]),
                                float(closes[c] - up_slope * rng)],
                    "lower_x": [idx[a], idx[b], idx[c]],
                    "lower_y": [float(closes[a] - dn_slope * rng),
                                float(closes[b]),
                                float(closes[c] + dn_slope * rng)],
                    "pivot": {"x": idx[b], "y": float(closes[b])},
                    "spread_start": float(spread_start),
                    "spread_end": float(spread_end),
                    "squeeze": float(squeeze),
                    "touch_up": touches_up,
                    "touch_dn": touches_dn})
        return pennants

    # ***********************************************************************************************************************
    # Functionname:       PatternAdapters._validate_ohlc_dataframe(ohlc: pd.DataFrame)
    #
    # @brief              Validate OHLC DataFrame schema.
    # @pre                Input object should be a DataFrame
    # @post               Raises ValueError for invalid OHLC layout
    # @param[in]          ohlc: Candidate OHLC DataFrame
    # @param[out]         result: None
    #
    # @callsequence       @startuml
    #                     title PatternAdapters._validate_ohlc_dataframe
    #
    #                     start
    #                     :Check ohlc is pandas DataFrame;
    #                     if (not DataFrame?) then (yes)
    #                       :Raise ValueError;
    #                       stop
    #                     endif
    #                     :Check required OHLC columns exist;
    #                     if (missing cols?) then (yes)
    #                       :Raise ValueError;
    #                       stop
    #                     endif
    #                     end
    #                     @enduml
    #
    # @InOutCorrelation   As described in UML diagram
    # @traceability
    # ***********************************************************************************************************************
    @staticmethod
    def _validate_ohlc_dataframe(ohlc: pd.DataFrame) -> None:
        required_cols = {"open", "high", "low", "close"}
        if not isinstance(ohlc, pd.DataFrame):
            raise ValueError("ohlc must be a pandas DataFrame.")
        if not required_cols.issubset(set(ohlc.columns)):
            raise ValueError("ohlc must include columns: open, high, low, close.")

# ---------------------------------------------------------------------------
# Backward-compatible module-level wrappers
# ---------------------------------------------------------------------------
def build_head_shoulders_struct(ohlc: pd.DataFrame, order: int = 10, rr: float = 1.8) -> dict[str, list[dict]]:
    return PatternAdapters.build_head_shoulders_struct(ohlc, order=order, rr=rr)

def build_pennants_struct(ohlc: pd.DataFrame, lookback: int = 60, min_touch: int = 3, max_squeeze: float = 0.35, rng: int = 40) -> list[dict]:
    return PatternAdapters.build_pennants_struct(ohlc, lookback=lookback, min_touch=min_touch, max_squeeze=max_squeeze, rng=rng)

def compute_candlestick_patterns(ohlc: pd.DataFrame, doji_threshold: float = 0.1, shadow_ratio: float = 2.0) -> pd.DataFrame:
    return PatternAdapters.compute_candlestick_patterns(ohlc, doji_threshold=doji_threshold, shadow_ratio=shadow_ratio)

__all__ = ["PatternAdapters", "build_head_shoulders_struct", "build_pennants_struct", "compute_candlestick_patterns"]
