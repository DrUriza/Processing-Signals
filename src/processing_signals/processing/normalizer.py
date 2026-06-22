from __future__ import annotations

from typing import Any
import pandas as pd


class Normalizer:
    """
    Processing layer: converts raw payloads to canonical dataframes/summaries.

    This module does not calculate indicators. It only standardizes the shape.
    """

    def normalize(self, payload: dict[str, Any], detected: dict[str, Any]) -> dict[str, Any]:
        data_type = detected["data_type"]

        if data_type == "candlestick":
            return self._normalize_candlesticks(payload, detected)

        if data_type == "orderbook_conventional":
            return self._normalize_orderbook(payload, detected)

        if data_type == "orderbook_large_trades":
            return self._normalize_large_trades(payload, detected)

        if data_type == "orderbook_whale_orders":
            return self._normalize_whale_orders(payload, detected)

        if data_type == "manifest":
            return {
                "kind": "manifest",
                "summary": payload.get("summary", {}),
                "raw": payload,
            }

        return {
            "kind": "unknown",
            "summary": {"error": "unknown data type"},
            "raw": payload,
        }

    def _normalize_candlesticks(self, payload: dict[str, Any], detected: dict[str, Any]) -> dict[str, Any]:
        candles = payload.get("candles", [])
        df = pd.DataFrame(candles)
        if df.empty:
            raise ValueError("Candlestick payload has no candle records.")

        rename_map = {
            "timestamp_utc": "timestamp",
            "volume_btc": "volume",
            "volume_usdt": "notional_volume",
        }
        df = df.rename(columns=rename_map)

        required = ["timestamp", "open", "high", "low", "close", "volume"]
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"Candlestick data missing required columns: {missing}")

        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        for col in ["open", "high", "low", "close", "volume", "notional_volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.sort_values("timestamp").reset_index(drop=True)

        return {
            "kind": "candlestick",
            "symbol": detected.get("symbol") or self._first_non_null(df, "symbol"),
            "timeframe": detected.get("timeframe") or self._first_non_null(df, "timeframe"),
            "dataframe": df,
            "summary": {
                "kind": "candlestick",
                "rows": int(len(df)),
                "symbol": detected.get("symbol") or self._first_non_null(df, "symbol"),
                "timeframe": detected.get("timeframe") or self._first_non_null(df, "timeframe"),
                "start": df["timestamp"].min().isoformat() if not df["timestamp"].isna().all() else None,
                "end": df["timestamp"].max().isoformat() if not df["timestamp"].isna().all() else None,
                "columns": list(df.columns),
            },
        }

    def _normalize_orderbook(self, payload: dict[str, Any], detected: dict[str, Any]) -> dict[str, Any]:
        bids = pd.DataFrame(payload.get("bids", []))
        asks = pd.DataFrame(payload.get("asks", []))
        if bids.empty or asks.empty:
            raise ValueError("Order book payload requires non-empty bids and asks.")

        for df in [bids, asks]:
            for col in ["level", "price", "quantity_btc", "notional_usdt"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

        bids = bids.sort_values("price", ascending=False).reset_index(drop=True)
        asks = asks.sort_values("price", ascending=True).reset_index(drop=True)

        best_bid = float(bids["price"].iloc[0])
        best_ask = float(asks["price"].iloc[0])

        return {
            "kind": "orderbook_conventional",
            "symbol": detected.get("symbol"),
            "timeframe": None,
            "bids": bids,
            "asks": asks,
            "summary": {
                "kind": "orderbook_conventional",
                "rows": int(len(bids) + len(asks)),
                "bid_levels": int(len(bids)),
                "ask_levels": int(len(asks)),
                "best_bid": best_bid,
                "best_ask": best_ask,
                "spread": best_ask - best_bid,
                "timestamp": detected.get("timestamp_utc"),
            },
        }

    def _normalize_large_trades(self, payload: dict[str, Any], detected: dict[str, Any]) -> dict[str, Any]:
        buy = pd.DataFrame(payload.get("large_buy_trades", []))
        sell = pd.DataFrame(payload.get("large_sell_trades", []))
        df = pd.concat([buy, sell], ignore_index=True)
        if df.empty:
            raise ValueError("Large trades payload has no records.")

        df["event_type"] = "large_trade"
        df["timestamp"] = pd.to_datetime(df.get("timestamp_utc"), utc=True, errors="coerce")
        snapshot_time = pd.to_datetime(detected.get("timestamp_utc"), utc=True, errors="coerce")

        if pd.notna(snapshot_time):
            df["age_seconds"] = (snapshot_time - df["timestamp"]).dt.total_seconds()

        for col in ["price", "quantity_btc", "notional_usdt", "age_seconds"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.sort_values("timestamp").reset_index(drop=True)

        return {
            "kind": "orderbook_large_trades",
            "symbol": detected.get("symbol"),
            "timeframe": None,
            "events": df,
            "summary": {
                "kind": "orderbook_large_trades",
                "rows": int(len(df)),
                "buy_rows": int((df["side"] == "buy").sum()) if "side" in df else 0,
                "sell_rows": int((df["side"] == "sell").sum()) if "side" in df else 0,
                "timestamp": detected.get("timestamp_utc"),
            },
        }

    def _normalize_whale_orders(self, payload: dict[str, Any], detected: dict[str, Any]) -> dict[str, Any]:
        buy = pd.DataFrame(payload.get("whale_buy_orders", []))
        sell = pd.DataFrame(payload.get("whale_sell_orders", []))
        df = pd.concat([buy, sell], ignore_index=True)
        if df.empty:
            raise ValueError("Whale orders payload has no records.")

        df["event_type"] = "whale_order"
        df["placed_at"] = pd.to_datetime(df.get("placed_at_utc"), utc=True, errors="coerce")
        df["snapshot_time"] = pd.to_datetime(df.get("snapshot_time_utc"), utc=True, errors="coerce")

        if "active_duration_seconds" not in df.columns:
            df["active_duration_seconds"] = (df["snapshot_time"] - df["placed_at"]).dt.total_seconds()

        for col in ["price", "quantity_btc", "notional_usdt", "active_duration_seconds", "active_duration_minutes"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.sort_values("placed_at").reset_index(drop=True)

        return {
            "kind": "orderbook_whale_orders",
            "symbol": detected.get("symbol"),
            "timeframe": None,
            "events": df,
            "summary": {
                "kind": "orderbook_whale_orders",
                "rows": int(len(df)),
                "buy_rows": int((df["side"] == "buy").sum()) if "side" in df else 0,
                "sell_rows": int((df["side"] == "sell").sum()) if "side" in df else 0,
                "timestamp": detected.get("timestamp_utc"),
            },
        }

    @staticmethod
    def _first_non_null(df: pd.DataFrame, column: str) -> Any:
        if column not in df.columns:
            return None
        values = df[column].dropna()
        return values.iloc[0] if not values.empty else None
