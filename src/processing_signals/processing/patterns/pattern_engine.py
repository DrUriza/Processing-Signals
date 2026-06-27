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

        if kind == "mining_network_health":
            return self._detect_mining_network_regimes(normalized, math_result)

        if kind == "onchain_holder_behavior":
            return self._detect_onchain_holder_regimes(normalized, math_result)

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

    def _detect_mining_network_regimes(self, normalized: dict[str, Any], math_result: dict[str, Any]) -> dict[str, Any]:
        df: pd.DataFrame = normalized["dataframe"]
        if df.empty:
            return {}

        last = df.iloc[-1]
        previous = df.iloc[-2] if len(df) > 1 else None
        regimes: list[str] = []

        miner_ratio = self._number(last.get("miner_ratio"))
        miner_netflow = self._number(last.get("miner_netflow_btc"))
        miner_inflow = self._number(last.get("miner_inflow_btc"))
        miner_outflow = self._number(last.get("miner_outflow_btc"))
        hash_rate = self._number(last.get("hash_rate", last.get("hashrate")))
        difficulty = self._number(last.get("difficulty"))

        if miner_ratio is not None and miner_ratio >= 1.2:
            regimes.append("miner_pressure_high")
        if miner_netflow is not None and miner_netflow < 0:
            regimes.append("miner_distribution")
        if miner_netflow is not None and miner_netflow > 0:
            regimes.append("miner_accumulation")
        if miner_outflow is not None and miner_inflow is not None and miner_outflow > miner_inflow:
            regimes.append("miner_distribution")

        if previous is not None:
            prev_hash_rate = self._number(previous.get("hash_rate", previous.get("hashrate")))
            prev_difficulty = self._number(previous.get("difficulty"))
            if hash_rate is not None and prev_hash_rate not in {None, 0}:
                hash_rate_change = (hash_rate - prev_hash_rate) / abs(prev_hash_rate)
                if hash_rate_change >= 0.02:
                    regimes.append("hash_rate_expansion")
                if hash_rate_change <= -0.02:
                    regimes.append("hash_rate_contraction")
            if difficulty is not None and prev_difficulty not in {None, 0} and hash_rate is not None and prev_hash_rate not in {None, 0}:
                difficulty_change = (difficulty - prev_difficulty) / abs(prev_difficulty)
                hash_rate_change = (hash_rate - prev_hash_rate) / abs(prev_hash_rate)
                if difficulty_change > 0 and hash_rate_change < 0:
                    regimes.append("network_stress")

        if miner_ratio is not None and miner_ratio >= 1.5:
            regimes.append("network_stress")

        return {"regimes": sorted(set(regimes))}

    def _detect_onchain_holder_regimes(self, normalized: dict[str, Any], math_result: dict[str, Any]) -> dict[str, Any]:
        df: pd.DataFrame = normalized["dataframe"]
        if df.empty:
            return {}

        last = df.iloc[-1]
        previous = df.iloc[-2] if len(df) > 1 else None
        regimes: list[str] = []

        accumulation_score = self._number(last.get("accumulation_score"))
        distribution_score = self._number(last.get("distribution_score"))
        exchange_balance = self._number(last.get("exchange_balance_btc"))
        mvrv = self._number(last.get("mvrv"))
        sopr = self._number(last.get("sopr"))
        lth_supply = self._number(last.get("long_term_holder_supply"))
        sth_supply = self._number(last.get("short_term_holder_supply"))

        if accumulation_score is not None and accumulation_score >= 0.6:
            regimes.append("accumulation_regime")
        if distribution_score is not None and distribution_score >= 0.6:
            regimes.append("distribution_regime")
        if lth_supply is not None and sth_supply is not None and lth_supply > sth_supply:
            regimes.append("holder_confidence_high")
        if mvrv is not None and mvrv >= 2.5:
            regimes.append("profit_taking_risk")
        if sopr is not None and sopr >= 1.05:
            regimes.append("profit_taking_risk")
        if mvrv is not None and mvrv <= 0.9:
            regimes.append("capitulation_risk")
        if sopr is not None and sopr < 1:
            regimes.append("capitulation_risk")

        if previous is not None:
            prev_exchange_balance = self._number(previous.get("exchange_balance_btc"))
            if exchange_balance is not None and prev_exchange_balance not in {None, 0}:
                balance_change = (exchange_balance - prev_exchange_balance) / abs(prev_exchange_balance)
                if balance_change >= 0.005:
                    regimes.append("exchange_balance_rising")
                if balance_change <= -0.005:
                    regimes.append("exchange_balance_falling")

        return {"regimes": sorted(set(regimes))}

    @staticmethod
    def _number(value: Any) -> float | None:
        try:
            parsed = pd.to_numeric(value, errors="coerce")
        except (TypeError, ValueError):
            return None
        if pd.isna(parsed):
            return None
        return float(parsed)
