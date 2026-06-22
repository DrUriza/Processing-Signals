from __future__ import annotations

from typing import Any


class OutputClassifier:
    """
    Classification layer.

    It does not calculate indicators. It only decides which payload families
    are useful for HMI, ML, and advanced algorithms.
    """

    def classify(
        self,
        detected: dict[str, Any],
        normalized: dict[str, Any],
        math_result: dict[str, Any],
        patterns: dict[str, Any],
        decision: dict[str, Any],
    ) -> dict[str, Any]:
        targets = decision.get("targets", {})
        kind = normalized.get("kind")

        routes = {
            "hmi": {"enabled": bool(targets.get("hmi")), "chart_type": None, "panels": [], "overlays": []},
            "ml": {"enabled": bool(targets.get("ml")), "include_feature_vector": False, "include_patterns": False},
            "advanced_algorithms": {
                "enabled": bool(targets.get("advanced_algorithms")),
                "preserve_time_series": False,
                "preserve_event_sequence": False,
                "suggested_modules": [],
            },
        }

        if kind == "candlestick":
            routes["hmi"].update(
                {
                    "chart_type": "candlestick",
                    "overlays": ["ema_20", "ema_50", "bollinger_bands", "vwap"],
                    "panels": ["rsi_14", "macd", "atr_14", "returns_rolling_kurtosis_30"],
                }
            )
            routes["ml"].update({"include_feature_vector": True, "include_patterns": True})
            routes["advanced_algorithms"].update(
                {
                    "preserve_time_series": True,
                    "suggested_modules": ["fourier", "wavelet", "kalman", "tda"],
                }
            )

        elif kind == "orderbook_conventional":
            routes["hmi"].update(
                {
                    "chart_type": "orderbook_depth",
                    "overlays": ["bid_wall_score", "ask_wall_score"],
                    "panels": ["orderbook_imbalance_total", "spread_bps"],
                }
            )
            routes["ml"].update({"include_feature_vector": True, "include_patterns": True})
            routes["advanced_algorithms"].update(
                {
                    "preserve_event_sequence": False,
                    "suggested_modules": ["liquidity_map", "orderbook_pressure"],
                }
            )

        elif kind in {"orderbook_large_trades", "orderbook_whale_orders"}:
            routes["hmi"].update(
                {
                    "chart_type": "event_timeline",
                    "overlays": ["buy_flow", "sell_flow"],
                    "panels": ["flow_imbalance", "notional_usdt", "age_seconds"],
                }
            )
            routes["ml"].update({"include_feature_vector": True, "include_patterns": True})
            routes["advanced_algorithms"].update(
                {
                    "preserve_event_sequence": True,
                    "suggested_modules": ["event_clustering", "flow_imbalance", "whale_ttl_analysis"],
                }
            )

        elif kind == "manifest":
            routes["hmi"].update({"chart_type": "metadata_summary"})

        return routes
