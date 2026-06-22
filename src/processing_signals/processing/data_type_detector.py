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

    def detect(self, payload: dict[str, Any], source_name: str = "") -> dict[str, Any]:
        metadata = payload.get("metadata", {}) if isinstance(payload.get("metadata", {}), dict) else {}
        metadata_type = str(metadata.get("data_type", "")).lower()
        order_book_type = str(metadata.get("order_book_type", "")).lower()

        if "candles" in payload or metadata_type in {"candlesticks", "ohlcv", "candles"}:
            return self._detected("candlestick", "ohlcv", metadata, source_name)

        if {"bids", "asks"}.issubset(payload.keys()) or order_book_type == "conventional":
            return self._detected("orderbook_conventional", "book_snapshot", metadata, source_name)

        if {"large_buy_trades", "large_sell_trades"}.intersection(payload.keys()) or order_book_type == "large_trades":
            return self._detected("orderbook_large_trades", "event_list", metadata, source_name)

        if {"whale_buy_orders", "whale_sell_orders"}.intersection(payload.keys()) or order_book_type == "whale_orders":
            return self._detected("orderbook_whale_orders", "event_list_with_ttl", metadata, source_name)

        if "files" in payload and "summary" in payload:
            return self._detected("manifest", "metadata", metadata, source_name)

        # Shape-based fallback for future APIs.
        if self._looks_like_candles(payload):
            return self._detected("candlestick", "ohlcv", metadata, source_name, confidence="medium")

        return self._detected("unknown", "unknown", metadata, source_name, confidence="low")

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
