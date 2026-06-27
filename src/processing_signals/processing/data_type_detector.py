from __future__ import annotations

from typing import Any


class DataTypeDetector:
    """
    Processing layer: detects the canonical data type.

    This is intentionally rule-based first, because incoming exchange/API JSONs are
    not always stable. Later this can become a schema registry.
    """

    CANDLE_KEYS = {"open", "high", "low", "close"}
    BOOK_LEVEL_KEYS = {"level", "side", "price"}
    TRADE_KEYS = {"side", "price", "quantity_btc", "notional_usdt"}
    WHALE_KEYS = {"placed_at_utc", "snapshot_time_utc", "active_duration_seconds"}
    MINING_NETWORK_TYPES = {
        "miner_ratio",
        "hash_rate",
        "miner_flows",
        "miner_inflow_outflow",
        "mining_network_health",
    }
    ONCHAIN_HOLDER_TYPES = {
        "glassnode_metrics",
        "holder_cohorts",
        "holder_behavior",
        "onchain_metrics",
        "onchain_holder_behavior",
    }
    MINING_NETWORK_FIELDS = {
        "miner_ratio",
        "hash_rate",
        "hashrate",
        "miner_inflow_btc",
        "miner_outflow_btc",
        "miner_netflow_btc",
        "difficulty",
    }
    ONCHAIN_HOLDER_FIELDS = {
        "mvrv",
        "nvt",
        "sopr",
        "long_term_holder_supply",
        "short_term_holder_supply",
        "accumulation_score",
        "distribution_score",
        "exchange_balance_btc",
    }

    def detect(self, payload: dict[str, Any], source_name: str = "") -> dict[str, Any]:
        metadata = payload.get("metadata", {}) if isinstance(payload.get("metadata", {}), dict) else {}
        metadata_type = str(metadata.get("data_type", "")).lower()
        order_book_type = str(metadata.get("order_book_type", "")).lower()
        payload_keys = self._payload_keys(payload)

        if "candles" in payload or metadata_type in {"candlesticks", "ohlcv", "candles"}:
            return self._detected("candlestick", "ohlcv", metadata, source_name)

        if (
            metadata_type in self.MINING_NETWORK_TYPES
            or any(token in metadata_type for token in ["miner", "mining", "hash_rate", "hashrate", "difficulty", "network_health"])
            or self.MINING_NETWORK_FIELDS.intersection(payload_keys)
        ):
            return self._detected("mining_network_health", "time_series", metadata, source_name)

        if (
            metadata_type in self.ONCHAIN_HOLDER_TYPES
            or any(
                token in metadata_type
                for token in ["onchain", "on_chain", "glassnode", "holder", "cohort", "mvrv", "nvt", "sopr"]
            )
            or self.ONCHAIN_HOLDER_FIELDS.intersection(payload_keys)
        ):
            return self._detected("onchain_holder_behavior", "time_series", metadata, source_name)

        if "cvd" in metadata_type:
            return self._detected("cvd", "time_series", metadata, source_name)

        if any(token in metadata_type for token in ["etf", "exchange_flow", "netflow", "inflow", "outflow"]):
            return self._detected(metadata_type, "time_series", metadata, source_name)

        if "liquidation" in metadata_type:
            return self._detected(metadata_type, "time_series", metadata, source_name)

        if "open_interest" in metadata_type or metadata_type == "oi":
            return self._detected(metadata_type, "time_series", metadata, source_name)

        if "long_short" in metadata_type:
            return self._detected(metadata_type, "time_series", metadata, source_name)

        if metadata_type == "orderbook_conventional" or order_book_type == "conventional":
            return self._detected("orderbook_conventional", "book_snapshot", metadata, source_name)

        if metadata_type == "orderbook_large_trades" or order_book_type == "large_trades":
            return self._detected("orderbook_large_trades", "event_list", metadata, source_name)

        if metadata_type == "orderbook_whale_orders" or order_book_type == "whale_orders":
            return self._detected("orderbook_whale_orders", "event_list_with_ttl", metadata, source_name)

        if {"bids", "asks"}.issubset(payload.keys()) or order_book_type == "conventional":
            return self._detected("orderbook_conventional", "book_snapshot", metadata, source_name)

        if {"large_buy_trades", "large_sell_trades"}.intersection(payload.keys()) or order_book_type == "large_trades":
            return self._detected("orderbook_large_trades", "event_list", metadata, source_name)

        if {"whale_buy_orders", "whale_sell_orders"}.intersection(payload.keys()) or order_book_type == "whale_orders":
            return self._detected("orderbook_whale_orders", "event_list_with_ttl", metadata, source_name)

        if "files" in payload and ("summary" in payload or "validation" in payload):
            return self._detected("manifest", "metadata", metadata, source_name)

        # Shape-based fallback for future APIs.
        if self._looks_like_candles(payload):
            return self._detected("candlestick", "ohlcv", metadata, source_name, confidence="medium")

        return self._detected("unknown", "unknown", metadata, source_name, confidence="low")

    def _payload_keys(self, payload: dict[str, Any]) -> set[str]:
        keys = {str(key).lower() for key in payload.keys()}
        records = payload.get("records")
        if isinstance(records, list) and records and isinstance(records[0], dict):
            keys.update(str(key).lower() for key in records[0].keys())
        return keys

    def _looks_like_candles(self, payload: dict[str, Any]) -> bool:
        candidate_lists = [value for value in payload.values() if isinstance(value, list) and value]
        for items in candidate_lists:
            first = items[0]
            if isinstance(first, dict) and self.CANDLE_KEYS.issubset(set(first.keys())):
                return True
        return False

    @staticmethod
    def _detected(
        data_type: str,
        canonical_type: str,
        metadata: dict[str, Any],
        source_name: str,
        confidence: str = "high",
    ) -> dict[str, Any]:
        return {
            "source_name": source_name,
            "data_type": data_type,
            "canonical_type": canonical_type,
            "confidence": confidence,
            "symbol": metadata.get("symbol"),
            "timeframe": metadata.get("timeframe"),
            "asset": metadata.get("asset"),
            "provider_source": metadata.get("source"),
            "timestamp_utc": metadata.get("timestamp_utc") or metadata.get("created_at_utc"),
            "metadata": metadata,
        }
