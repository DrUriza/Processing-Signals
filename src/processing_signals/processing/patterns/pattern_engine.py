from __future__ import annotations

from typing import Any

import pandas as pd

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
            patterns = self.candlestick_detector.detect(
                normalized=normalized,
                math_result=math_result,
                transforms=transforms,
                view_math=view_math,
            )
            self._add_candlestick_view_patterns(patterns, transforms, math_result, view_math)
            return patterns

        if kind == "orderbook_conventional":
            patterns = self.liquidity_detector.detect(
                normalized=normalized,
                math_result=math_result,
                transforms=transforms,
                view_math=view_math,
            )
            self._add_candlestick_view_patterns(patterns, transforms, math_result, view_math)
            return patterns

        if kind in {"orderbook_large_trades", "orderbook_whale_orders"}:
            patterns = self.event_detector.detect(
                normalized=normalized,
                math_result=math_result,
                transforms=transforms,
                view_math=view_math,
            )
            self._add_candlestick_view_patterns(patterns, transforms, math_result, view_math)
            return patterns

        if kind == "mining_network_health":
            patterns = self.mining_detector.detect(
                normalized=normalized,
                math_result=math_result,
                transforms=transforms,
                view_math=view_math,
            )
            self._add_candlestick_view_patterns(patterns, transforms, math_result, view_math)
            return patterns

        if kind == "onchain_holder_behavior":
            patterns = self.onchain_detector.detect(
                normalized=normalized,
                math_result=math_result,
                transforms=transforms,
                view_math=view_math,
            )
            self._add_candlestick_view_patterns(patterns, transforms, math_result, view_math)
            return patterns

        patterns: dict[str, Any] = {}
        self._add_candlestick_view_patterns(patterns, transforms, math_result, view_math)
        return patterns

    def _add_candlestick_view_patterns(
        self,
        patterns: dict[str, Any],
        transforms: dict[str, Any],
        math_result: dict[str, Any],
        view_math: dict[str, Any],
    ) -> None:
        view_patterns: dict[str, Any] = {}
        for view_name in ["candlestick_derived", "cvd_candlestick_derived"]:
            records = transforms.get(view_name, {}).get("records", [])
            if not records:
                continue
            detected = self.candlestick_detector.detect(
                normalized={"kind": view_name, "dataframe": pd.DataFrame(records)},
                math_result=math_result,
                transforms=transforms,
                view_math=view_math,
            )
            compact = self._compact_view_patterns(detected)
            if compact:
                view_patterns[view_name] = compact
        if view_patterns:
            patterns["view_patterns"] = view_patterns

    @staticmethod
    def _compact_view_patterns(patterns: dict[str, Any]) -> dict[str, Any]:
        if not patterns:
            return {}
        return {
            "pattern_groups": patterns.get("pattern_groups", {}),
            "pattern_summary": patterns.get("pattern_summary", {}),
            "candle_shape": patterns.get("candle_shape", {}),
            "pattern_inputs": patterns.get("pattern_inputs", {}),
        }
