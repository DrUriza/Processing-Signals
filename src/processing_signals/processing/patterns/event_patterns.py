from __future__ import annotations

from typing import Any


class EventPatternDetector:
    def detect(
        self,
        normalized: dict[str, Any],
        math_result: dict[str, Any],
        transforms: dict[str, Any] | None = None,
        view_math: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        transforms = transforms or {}
        micro = math_result.get("microstructure", {})
        kind = normalized.get("kind")
        patterns = []

        imbalance = micro.get("flow_imbalance")
        if imbalance is not None and imbalance > 0.4:
            patterns.append(self._pattern("buy_flow_dominance", "event_flow", "bullish", 0.72, "Event flow imbalance favors buy-side pressure."))
        if imbalance is not None and imbalance < -0.4:
            patterns.append(self._pattern("sell_flow_dominance", "event_flow", "bearish", 0.72, "Event flow imbalance favors sell-side pressure."))

        if kind == "orderbook_whale_orders":
            max_age = micro.get("whale_order_age_minutes_max")
            mean_age = micro.get("whale_order_age_minutes_mean")
            if max_age is not None and max_age > 180:
                patterns.append(self._pattern("old_whale_order_still_active", "event_flow", "neutral", 0.68, "Old whale order remains active beyond age threshold."))
            if mean_age is not None and mean_age > 60:
                patterns.append(self._pattern("persistent_whale_liquidity", "event_flow", "neutral", 0.70, "Average whale order age indicates persistent liquidity."))
                patterns.append(self._pattern("whale_liquidity_persistence", "event_flow", "neutral", 0.70, "Whale liquidity persistence threshold is active."))

        if kind == "orderbook_large_trades":
            max_age = micro.get("event_age_seconds_max")
            if max_age is not None and max_age < 600:
                patterns.append(self._pattern("recent_large_trade_cluster", "event_flow", "neutral", 0.70, "Large trade events are clustered in recent time."))
                patterns.append(self._pattern("large_trade_cluster", "event_flow", "neutral", 0.70, "Large trade cluster threshold is active."))

        event_candidates = transforms.get("event_list", {}).get("events", [])
        patterns = self._unique_patterns(patterns)
        return {
            "pattern_groups": {
                "candlestick_patterns": {},
                "liquidity_patterns": [],
                "event_patterns": patterns,
                "mining_patterns": [],
                "onchain_patterns": [],
            },
            "pattern_summary": self._pattern_summary(patterns),
            "pattern_inputs": {
                "uses_technical_indicators": False,
                "uses_statistics": bool(math_result.get("statistics")),
                "uses_statistical_regimes": bool(math_result.get("statistical_regimes")),
                "uses_regime_flags": bool(math_result.get("statistical_regimes", {}).get("regime_flags")),
                "uses_microstructure": bool(micro),
                "microstructure": sorted(micro.keys()),
                "event_candidates": len(event_candidates) if hasattr(event_candidates, "__len__") else 0,
            },
        }

    @staticmethod
    def _pattern(name: str, category: str, direction: str, confidence: float, reason: str) -> dict[str, Any]:
        return {
            "name": name,
            "category": category,
            "direction": direction,
            "confidence": max(0.0, min(1.0, confidence)),
            "timestamp": None,
            "start_index": None,
            "end_index": None,
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
