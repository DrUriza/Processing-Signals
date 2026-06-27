from __future__ import annotations

from typing import Any

import pandas as pd

from processing_signals.processing.math.statistics import coerce_numeric_series


class IndicatorDecisionEngine:
    """
    Processing layer: decides what should be calculated.

    This answers:
      - Should technical indicators be applied?
      - Should pure statistical metrics be applied?
      - Should microstructure metrics be applied?
      - Should pattern detection be applied?
      - Which output routes are needed?
    """

    def decide(self, detected: dict[str, Any], normalized: dict[str, Any]) -> dict[str, Any]:
        data_type = detected["data_type"]

        base = {
            "data_type": data_type,
            "canonical_type": detected.get("canonical_type"),
            "apply_technical_indicators": False,
            "apply_statistical_metrics": False,
            "apply_microstructure_metrics": False,
            "apply_patterns": False,
            "targets": {
                "hmi": False,
                "ml": False,
                "advanced_algorithms": False,
            },
            "notes": [],
        }

        if data_type == "candlestick":
            base.update(
                {
                    "apply_technical_indicators": True,
                    "apply_statistical_metrics": True,
                    "apply_microstructure_metrics": False,
                    "apply_patterns": True,
                    "targets": {"hmi": True, "ml": True, "advanced_algorithms": True},
                }
            )
            base["notes"].append("OHLCV supports technical indicators, statistics, and candle/time-series patterns.")

        elif data_type == "orderbook_conventional":
            base.update(
                {
                    "apply_technical_indicators": False,
                    "apply_statistical_metrics": True,
                    "apply_microstructure_metrics": True,
                    "apply_patterns": True,
                    "targets": {"hmi": True, "ml": True, "advanced_algorithms": True},
                }
            )
            base["notes"].append("Order book snapshot uses microstructure metrics, liquidity patterns, and ML features.")

        elif data_type == "orderbook_large_trades":
            base.update(
                {
                    "apply_technical_indicators": False,
                    "apply_statistical_metrics": True,
                    "apply_microstructure_metrics": True,
                    "apply_patterns": True,
                    "targets": {"hmi": True, "ml": True, "advanced_algorithms": True},
                }
            )
            base["notes"].append("Large trades are event-list data; use event statistics, imbalance, and flow patterns.")

        elif data_type == "orderbook_whale_orders":
            base.update(
                {
                    "apply_technical_indicators": False,
                    "apply_statistical_metrics": True,
                    "apply_microstructure_metrics": True,
                    "apply_patterns": True,
                    "targets": {"hmi": True, "ml": True, "advanced_algorithms": True},
                }
            )
            base["notes"].append("Whale orders include TTL/active duration; route to HMI, ML, and advanced algorithms.")

        elif data_type == "manifest":
            base["targets"]["hmi"] = True
            base["notes"].append("Manifest is metadata only; no indicators.")

        elif data_type in {"mining_network_health", "onchain_holder_behavior"} or self._has_numeric_records(normalized):
            base.update(
                {
                    "apply_technical_indicators": False,
                    "apply_statistical_metrics": True,
                    "apply_microstructure_metrics": False,
                    "apply_patterns": True,
                    "targets": {"hmi": True, "ml": True, "advanced_algorithms": True},
                }
            )
            base["notes"].append("No classic technical indicators applied; statistical metrics enabled.")

        else:
            base["notes"].append("Unknown type; no indicators applied.")

        return base

    @staticmethod
    def _has_numeric_records(normalized: dict[str, Any]) -> bool:
        for key in ("dataframe", "events", "bids", "asks"):
            value = normalized.get(key)
            if not isinstance(value, pd.DataFrame) or value.empty:
                continue
            for column in value.columns:
                if str(column).lower() in {
                    "timestamp",
                    "timestamp_utc",
                    "symbol",
                    "timeframe",
                    "family_key",
                    "data_type",
                    "source_subtype",
                    "provider",
                    "exchange",
                    "asset",
                    "base_asset",
                    "quote_asset",
                }:
                    continue
                numeric = coerce_numeric_series(value, column)
                if numeric.notna().any():
                    return True
        return False
