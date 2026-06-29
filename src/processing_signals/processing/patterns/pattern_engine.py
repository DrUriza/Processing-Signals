from __future__ import annotations

from typing import Any

from .candlestick_patterns import CandlestickPatternDetector
from .event_patterns import EventPatternDetector
from .liquidity_patterns import LiquidityPatternDetector
from .mining_patterns import MiningPatternDetector
from .onchain_patterns import OnchainPatternDetector


class PatternEngine:
    def __init__(self) -> None:
        self.candlestick_detector = CandlestickPatternDetector()
        self.liquidity_detector = LiquidityPatternDetector()
        self.event_detector = EventPatternDetector()
        self.mining_detector = MiningPatternDetector()
        self.onchain_detector = OnchainPatternDetector()

    def detect(
        self,
        normalized: dict[str, Any],
        math_result: dict[str, Any],
        decision: dict[str, Any],
        transforms: dict[str, Any] | None = None,
        view_math: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not decision.get("apply_patterns"):
            return {}

        kind = normalized.get("kind")
        transforms = transforms or {}
        view_math = view_math or {}

        if kind == "candlestick":
            return self.candlestick_detector.detect(
                normalized=normalized,
                math_result=math_result,
                transforms=transforms,
                view_math=view_math,
            )

        if kind == "orderbook_conventional":
            return self.liquidity_detector.detect(
                normalized=normalized,
                math_result=math_result,
                transforms=transforms,
                view_math=view_math,
            )

        if kind in {"orderbook_large_trades", "orderbook_whale_orders"}:
            return self.event_detector.detect(
                normalized=normalized,
                math_result=math_result,
                transforms=transforms,
                view_math=view_math,
            )

        if kind == "mining_network_health":
            return self.mining_detector.detect(
                normalized=normalized,
                math_result=math_result,
                transforms=transforms,
                view_math=view_math,
            )

        if kind == "onchain_holder_behavior":
            return self.onchain_detector.detect(
                normalized=normalized,
                math_result=math_result,
                transforms=transforms,
                view_math=view_math,
            )

        return {}
