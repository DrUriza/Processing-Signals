from __future__ import annotations

from typing import Any


class LiquidityPatternDetector:
    def detect(
        self,
        normalized: dict[str, Any],
        math_result: dict[str, Any],
        transforms: dict[str, Any] | None = None,
        view_math: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        transforms = transforms or {}
        micro = math_result.get("microstructure", {})
        regime_flags = math_result.get("statistical_regimes", {}).get("regime_flags", {})
        patterns = []

        imbalance = micro.get("orderbook_imbalance_total")
        bid_wall = micro.get("bid_wall_score")
        ask_wall = micro.get("ask_wall_score")
        spread_bps = micro.get("spread_bps")

        if imbalance is not None and imbalance > 0.2:
            patterns.append(self._pattern("bid_liquidity_dominance", "liquidity", "bullish", 0.72, "Orderbook imbalance favors bid liquidity."))
        if imbalance is not None and imbalance < -0.2:
            patterns.append(self._pattern("ask_liquidity_dominance", "liquidity", "bearish", 0.72, "Orderbook imbalance favors ask liquidity."))
        if bid_wall is not None and bid_wall > 0.65:
            patterns.append(self._pattern("bid_wall", "liquidity", "bullish", 0.76, "Bid wall score is above threshold."))
        if ask_wall is not None and ask_wall > 0.65:
            patterns.append(self._pattern("ask_wall", "liquidity", "bearish", 0.76, "Ask wall score is above threshold."))
        if spread_bps is not None and spread_bps > 10:
            patterns.append(self._pattern("wide_spread", "liquidity", "neutral", 0.68, "Spread is wider than the configured basis-point threshold."))

        event_candidates = transforms.get("event_list", {}).get("events", [])

        return {
            "pattern_groups": {
                "candlestick_patterns": {},
                "liquidity_patterns": patterns,
                "event_patterns": [],
                "mining_patterns": [],
                "onchain_patterns": [],
            },
            "pattern_summary": self._pattern_summary(patterns),
            "pattern_inputs": {
                "uses_technical_indicators": False,
                "uses_statistics": bool(math_result.get("statistics")),
                "uses_statistical_regimes": bool(math_result.get("statistical_regimes")),
                "uses_regime_flags": bool(regime_flags),
                "uses_microstructure": bool(micro),
                "microstructure": sorted(micro.keys()),
                "event_candidates": len(event_candidates) if isinstance(event_candidates, list) else 0,
                "regime_flags": regime_flags,
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
