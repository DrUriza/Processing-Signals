from __future__ import annotations

from typing import Any


class IndicatorDecisionEngine:
    """Control decisions for math and downstream routing; does not calculate math."""

    def decide(self, detected: dict[str, Any], normalized: dict[str, Any]) -> dict[str, Any]:
        kind = normalized.get("kind")
        data_type = detected.get("data_type")
        is_ohlc = self._is_ohlc_compatible(normalized)

        decision = {
            "apply_technical_indicators": False,
            "apply_statistics": True,
            "apply_statistical_regimes": True,
            "apply_patterns": True,
            "targets": {
                "hmi": True,
                "ml": True,
                "advanced_algorithms": True,
            },
        }

        if kind == "manifest" or data_type == "manifest":
            decision.update(
                {
                    "apply_statistics": False,
                    "apply_statistical_regimes": False,
                    "apply_patterns": False,
                    "targets": {
                        "hmi": True,
                        "ml": False,
                        "advanced_algorithms": False,
                    },
                }
            )
            return decision

        decision["apply_technical_indicators"] = bool(is_ohlc)
        return decision

    @staticmethod
    def _is_ohlc_compatible(normalized: dict[str, Any]) -> bool:
        dataframe = normalized.get("dataframe")
        if dataframe is None or not hasattr(dataframe, "columns"):
            return False
        return {"timestamp", "open", "high", "low", "close"}.issubset(set(dataframe.columns))


__all__ = ["IndicatorDecisionEngine"]
