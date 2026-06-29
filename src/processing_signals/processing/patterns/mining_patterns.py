from __future__ import annotations

from typing import Any

import pandas as pd


class MiningPatternDetector:
    def detect(
        self,
        normalized: dict[str, Any],
        math_result: dict[str, Any],
        transforms: dict[str, Any] | None = None,
        view_math: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        df = normalized.get("dataframe")
        if df is None or df.empty:
            return {}

        last = df.iloc[-1]
        last_index = len(df) - 1
        timestamp = last.get("timestamp")
        previous = df.iloc[-2] if len(df) > 1 else None
        patterns: list[dict[str, Any]] = []

        miner_ratio = self._number(last.get("miner_ratio"))
        miner_netflow = self._number(last.get("miner_netflow_btc"))
        miner_inflow = self._number(last.get("miner_inflow_btc"))
        miner_outflow = self._number(last.get("miner_outflow_btc"))
        hash_rate = self._number(last.get("hash_rate", last.get("hashrate")))
        difficulty = self._number(last.get("difficulty"))

        if miner_ratio is not None and miner_ratio >= 1.2:
            patterns.append(self._pattern("miner_pressure_high", "mining_regime", "bearish", 0.74, timestamp, last_index, "Miner ratio is above pressure threshold."))
        if miner_netflow is not None and miner_netflow < 0:
            patterns.append(self._pattern("miner_distribution", "mining_regime", "bearish", 0.70, timestamp, last_index, "Miner netflow is negative."))
        if miner_netflow is not None and miner_netflow > 0:
            patterns.append(self._pattern("miner_accumulation", "mining_regime", "bullish", 0.70, timestamp, last_index, "Miner netflow is positive."))
        if miner_outflow is not None and miner_inflow is not None and miner_outflow > miner_inflow:
            patterns.append(self._pattern("miner_distribution", "mining_regime", "bearish", 0.70, timestamp, last_index, "Miner outflow is greater than miner inflow."))

        if previous is not None:
            prev_hash_rate = self._number(previous.get("hash_rate", previous.get("hashrate")))
            prev_difficulty = self._number(previous.get("difficulty"))
            if hash_rate is not None and prev_hash_rate not in {None, 0}:
                hash_rate_change = (hash_rate - prev_hash_rate) / abs(prev_hash_rate)
                if hash_rate_change >= 0.02:
                    patterns.append(self._pattern("hash_rate_expansion", "mining_regime", "bullish", 0.68, timestamp, last_index, "Hash rate expanded versus previous sample."))
                if hash_rate_change <= -0.02:
                    patterns.append(self._pattern("hash_rate_contraction", "mining_regime", "bearish", 0.68, timestamp, last_index, "Hash rate contracted versus previous sample."))
            if difficulty is not None and prev_difficulty not in {None, 0} and hash_rate is not None and prev_hash_rate not in {None, 0}:
                difficulty_change = (difficulty - prev_difficulty) / abs(prev_difficulty)
                hash_rate_change = (hash_rate - prev_hash_rate) / abs(prev_hash_rate)
                if difficulty_change > 0 and hash_rate_change < 0:
                    patterns.append(self._pattern("network_stress", "mining_regime", "bearish", 0.76, timestamp, last_index, "Difficulty increased while hash rate contracted."))

        if miner_ratio is not None and miner_ratio >= 1.5:
            patterns.append(self._pattern("network_stress", "mining_regime", "bearish", 0.76, timestamp, last_index, "Miner ratio is above network stress threshold."))

        patterns = self._unique_patterns(patterns)
        return {
            "pattern_groups": {
                "candlestick_patterns": {},
                "liquidity_patterns": [],
                "event_patterns": [],
                "mining_patterns": patterns,
                "onchain_patterns": [],
            },
            "pattern_summary": self._pattern_summary(patterns),
            "pattern_inputs": {
                "uses_technical_indicators": False,
                "uses_statistics": bool(math_result.get("statistics")),
                "uses_statistical_regimes": bool(math_result.get("statistical_regimes")),
                "uses_regime_flags": bool(math_result.get("statistical_regimes", {}).get("regime_flags")),
                "uses_microstructure": False,
                "columns": [column for column in ["miner_ratio", "miner_netflow_btc", "miner_inflow_btc", "miner_outflow_btc", "hash_rate", "hashrate", "difficulty"] if column in df.columns],
            },
        }

    @staticmethod
    def _number(value: Any) -> float | None:
        try:
            parsed = pd.to_numeric(value, errors="coerce")
        except (TypeError, ValueError):
            return None
        if pd.isna(parsed):
            return None
        return float(parsed)

    @staticmethod
    def _pattern(name: str, category: str, direction: str, confidence: float, timestamp: Any, index: int, reason: str) -> dict[str, Any]:
        return {
            "name": name,
            "category": category,
            "direction": direction,
            "confidence": max(0.0, min(1.0, confidence)),
            "timestamp": timestamp,
            "start_index": index,
            "end_index": index,
            "reason": reason,
        }

    @staticmethod
    def _unique_patterns(patterns: list[dict[str, Any]]) -> list[dict[str, Any]]:
        unique: list[dict[str, Any]] = []
        seen: set[str] = set()
        for pattern in patterns:
            name = pattern["name"]
            if name in seen:
                continue
            seen.add(name)
            unique.append(pattern)
        return unique

    @staticmethod
    def _has_direction(patterns: list[dict[str, Any]], direction: str) -> bool:
        return any(pattern.get("direction") == direction for pattern in patterns)

    @staticmethod
    def _pattern_summary(patterns: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "total_patterns": len(patterns),
            "bullish_count": sum(1 for pattern in patterns if pattern.get("direction") == "bullish"),
            "bearish_count": sum(1 for pattern in patterns if pattern.get("direction") == "bearish"),
            "neutral_count": sum(1 for pattern in patterns if pattern.get("direction") == "neutral"),
            "high_confidence_count": sum(1 for pattern in patterns if pattern.get("confidence", 0) >= 0.75),
            "active_categories": sorted({pattern["category"] for pattern in patterns}),
        }
