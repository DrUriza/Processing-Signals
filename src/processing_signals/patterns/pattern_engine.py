from __future__ import annotations

from typing import Any

import pandas as pd


class PatternEngine:
    """
    Pattern processing layer.

    First version:
      - candlestick patterns
      - statistical regime patterns
      - liquidity/order-flow patterns
    """

    def detect(self, normalized: dict[str, Any], math_result: dict[str, Any], decision: dict[str, Any]) -> dict[str, Any]:
        if not decision.get("apply_patterns"):
            return {}

        kind = normalized.get("kind")

        if kind == "candlestick":
            return self._detect_candlestick_patterns(normalized, math_result)

        if kind == "orderbook_conventional":
            return self._detect_orderbook_patterns(math_result)

        if kind in {"orderbook_large_trades", "orderbook_whale_orders"}:
            return self._detect_event_patterns(normalized, math_result)

        return {}

    def _detect_candlestick_patterns(self, normalized: dict[str, Any], math_result: dict[str, Any]) -> dict[str, Any]:
        df: pd.DataFrame = normalized["dataframe"]
        if df.empty:
            return {}

        last = df.iloc[-1]
        body = abs(float(last["close"]) - float(last["open"]))
        candle_range = float(last["high"]) - float(last["low"])
        upper_wick = float(last["high"]) - max(float(last["open"]), float(last["close"]))
        lower_wick = min(float(last["open"]), float(last["close"])) - float(last["low"])
        body_ratio = body / candle_range if candle_range else 0

        patterns = []
        if candle_range and body_ratio < 0.1:
            patterns.append("doji")
        if candle_range and lower_wick > 2 * body and upper_wick < body:
            patterns.append("hammer_like")
        if float(last["close"]) > float(last["open"]):
            patterns.append("bullish_candle")
        elif float(last["close"]) < float(last["open"]):
            patterns.append("bearish_candle")

        stat_last = math_result.get("statistical", {}).get("last", {})
        statistical_regimes = []

        z30 = stat_last.get("rolling_zscore_30")
        kurt30 = stat_last.get("returns_rolling_kurtosis_30")
        skew30 = stat_last.get("returns_rolling_skewness_30")
        std30 = stat_last.get("returns_rolling_std_30")

        if z30 is not None and abs(z30) >= 2:
            statistical_regimes.append("mean_reversion_extreme")
        if kurt30 is not None and kurt30 >= 3:
            statistical_regimes.append("fat_tail_regime")
        if skew30 is not None and skew30 <= -1:
            statistical_regimes.append("negative_skew_risk")
        if skew30 is not None and skew30 >= 1:
            statistical_regimes.append("positive_skew_pressure")
        if std30 is not None and std30 > 0:
            statistical_regimes.append("volatility_active")

        return {
            "candlestick": patterns,
            "candle_shape": {
                "body": body,
                "range": candle_range,
                "upper_wick": upper_wick,
                "lower_wick": lower_wick,
                "body_ratio": body_ratio,
            },
            "statistical_regime": statistical_regimes,
        }

    def _detect_orderbook_patterns(self, math_result: dict[str, Any]) -> dict[str, Any]:
        micro = math_result.get("microstructure", {})
        patterns = []

        imbalance = micro.get("orderbook_imbalance_total")
        bid_wall = micro.get("bid_wall_score")
        ask_wall = micro.get("ask_wall_score")
        spread_bps = micro.get("spread_bps")

        if imbalance is not None and imbalance > 0.2:
            patterns.append("bid_liquidity_dominance")
        if imbalance is not None and imbalance < -0.2:
            patterns.append("ask_liquidity_dominance")
        if bid_wall is not None and bid_wall > 0.65:
            patterns.append("bid_wall")
        if ask_wall is not None and ask_wall > 0.65:
            patterns.append("ask_wall")
        if spread_bps is not None and spread_bps > 10:
            patterns.append("wide_spread")

        return {"liquidity_patterns": patterns}

    def _detect_event_patterns(self, normalized: dict[str, Any], math_result: dict[str, Any]) -> dict[str, Any]:
        micro = math_result.get("microstructure", {})
        kind = normalized.get("kind")
        patterns = []

        imbalance = micro.get("flow_imbalance")
        if imbalance is not None and imbalance > 0.4:
            patterns.append("buy_flow_dominance")
        if imbalance is not None and imbalance < -0.4:
            patterns.append("sell_flow_dominance")

        if kind == "orderbook_whale_orders":
            max_age = micro.get("whale_order_age_minutes_max")
            mean_age = micro.get("whale_order_age_minutes_mean")
            if max_age is not None and max_age > 180:
                patterns.append("old_whale_order_still_active")
            if mean_age is not None and mean_age > 60:
                patterns.append("persistent_whale_liquidity")

        if kind == "orderbook_large_trades":
            max_age = micro.get("event_age_seconds_max")
            if max_age is not None and max_age < 600:
                patterns.append("recent_large_trade_cluster")

        return {"event_patterns": patterns}
