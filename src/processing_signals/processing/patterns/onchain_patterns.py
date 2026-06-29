from __future__ import annotations

from typing import Any

import pandas as pd


class OnchainPatternDetector:
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

        accumulation_score = self._number(last.get("accumulation_score"))
        distribution_score = self._number(last.get("distribution_score"))
        exchange_balance = self._number(last.get("exchange_balance_btc"))
        mvrv = self._number(last.get("mvrv"))
        sopr = self._number(last.get("sopr"))
        lth_supply = self._number(last.get("long_term_holder_supply"))
        sth_supply = self._number(last.get("short_term_holder_supply"))

        if accumulation_score is not None and accumulation_score >= 0.6:
            patterns.append(self._pattern("accumulation_regime", "onchain_regime", "bullish", 0.72, timestamp, last_index, "Accumulation score is above threshold."))
        if distribution_score is not None and distribution_score >= 0.6:
            patterns.append(self._pattern("distribution_regime", "onchain_regime", "bearish", 0.72, timestamp, last_index, "Distribution score is above threshold."))
        if lth_supply is not None and sth_supply is not None and lth_supply > sth_supply:
            patterns.append(self._pattern("holder_confidence_high", "onchain_regime", "bullish", 0.68, timestamp, last_index, "Long-term holder supply is greater than short-term holder supply."))
        if mvrv is not None and mvrv >= 2.5:
            patterns.append(self._pattern("profit_taking_risk", "onchain_regime", "bearish", 0.70, timestamp, last_index, "MVRV is above profit-taking threshold."))
        if sopr is not None and sopr >= 1.05:
            patterns.append(self._pattern("profit_taking_risk", "onchain_regime", "bearish", 0.70, timestamp, last_index, "SOPR is above profit-taking threshold."))
        if mvrv is not None and mvrv <= 0.9:
            patterns.append(self._pattern("capitulation_risk", "onchain_regime", "bearish", 0.70, timestamp, last_index, "MVRV is below capitulation threshold."))
        if sopr is not None and sopr < 1:
            patterns.append(self._pattern("capitulation_risk", "onchain_regime", "bearish", 0.70, timestamp, last_index, "SOPR is below capitulation threshold."))

        if previous is not None:
            prev_exchange_balance = self._number(previous.get("exchange_balance_btc"))
            if exchange_balance is not None and prev_exchange_balance not in {None, 0}:
                balance_change = (exchange_balance - prev_exchange_balance) / abs(prev_exchange_balance)
                if balance_change >= 0.005:
                    patterns.append(self._pattern("exchange_balance_rising", "onchain_regime", "bearish", 0.66, timestamp, last_index, "Exchange balance increased versus previous sample."))
                if balance_change <= -0.005:
                    patterns.append(self._pattern("exchange_balance_falling", "onchain_regime", "bullish", 0.66, timestamp, last_index, "Exchange balance decreased versus previous sample."))

        patterns = self._unique_patterns(patterns)
        return {
            "pattern_groups": {
                "candlestick_patterns": {},
                "liquidity_patterns": [],
                "event_patterns": [],
                "mining_patterns": [],
                "onchain_patterns": patterns,
            },
            "pattern_summary": self._pattern_summary(patterns),
            "pattern_inputs": {
                "uses_technical_indicators": False,
                "uses_statistics": bool(math_result.get("statistics")),
                "uses_statistical_regimes": bool(math_result.get("statistical_regimes")),
                "uses_regime_flags": bool(math_result.get("statistical_regimes", {}).get("regime_flags")),
                "uses_microstructure": False,
                "columns": [
                    column
                    for column in [
                        "accumulation_score",
                        "distribution_score",
                        "exchange_balance_btc",
                        "mvrv",
                        "sopr",
                        "long_term_holder_supply",
                        "short_term_holder_supply",
                    ]
                    if column in df.columns
                ],
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
    def _pattern_summary(patterns: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "total_patterns": len(patterns),
            "bullish_count": sum(1 for pattern in patterns if pattern.get("direction") == "bullish"),
            "bearish_count": sum(1 for pattern in patterns if pattern.get("direction") == "bearish"),
            "neutral_count": sum(1 for pattern in patterns if pattern.get("direction") == "neutral"),
            "high_confidence_count": sum(1 for pattern in patterns if pattern.get("confidence", 0) >= 0.75),
            "active_categories": sorted({pattern["category"] for pattern in patterns}),
        }
