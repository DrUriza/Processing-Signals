from __future__ import annotations

from typing import Any
import pandas as pd


class Normalizer:
    """
    Processing layer: converts raw payloads to canonical dataframes/summaries.

    This module does not calculate indicators. It only standardizes the shape.
    """

    METRIC_DATA_TYPES = {
        "miner_ratio",
        "hash_rate",
        "miner_flows",
        "miner_inflow_outflow",
        "mining_network_health",
        "glassnode_metrics",
        "holder_cohorts",
        "holder_behavior",
        "onchain_metrics",
        "onchain_holder_behavior",
    }

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

        if data_type in self.METRIC_DATA_TYPES:
            return self._normalize_metric_records(payload, detected)

        if "records" in payload:
            return self._normalize_records(payload, detected)

        return {
            "kind": "unknown",
            "summary": {"error": "unknown data type"},
            "raw": payload,
        }

    def _normalize_candlesticks(self, payload: dict[str, Any], detected: dict[str, Any]) -> dict[str, Any]:
        candles = payload.get("candles") or payload.get("records", [])
        df = pd.DataFrame(candles)
        if df.empty:
            raise ValueError("Candlestick payload has no candle records.")

        rename_map = {
            "timestamp_utc": "timestamp",
            "volume_btc": "volume",
            "volume_usdt": "notional_volume",
        }
        df = df.rename(columns=rename_map)
        df = self._coalesce_duplicate_columns(df)

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
        records = payload.get("records")
        if isinstance(records, list) and records and isinstance(records[-1], dict) and {"bids", "asks"}.issubset(records[-1]):
            return self._normalize_orderbook_records(records, detected)

        bids = pd.DataFrame(payload.get("bids", []))
        asks = pd.DataFrame(payload.get("asks", []))
        if bids.empty or asks.empty:
            raise ValueError("Order book payload requires non-empty bids and asks.")

        bids = self._normalize_notional_aliases(bids)
        asks = self._normalize_notional_aliases(asks)
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
        if "records" in payload:
            return self._normalize_event_records(payload, detected, "orderbook_large_trades", "large_trade")

        buy = pd.DataFrame(payload.get("large_buy_trades", []))
        sell = pd.DataFrame(payload.get("large_sell_trades", []))
        df = pd.concat([buy, sell], ignore_index=True)
        if df.empty:
            raise ValueError("Large trades payload has no records.")

        df = self._normalize_notional_aliases(df)
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
        if "records" in payload:
            return self._normalize_event_records(payload, detected, "orderbook_whale_orders", "whale_order")

        buy = pd.DataFrame(payload.get("whale_buy_orders", []))
        sell = pd.DataFrame(payload.get("whale_sell_orders", []))
        df = pd.concat([buy, sell], ignore_index=True)
        if df.empty:
            raise ValueError("Whale orders payload has no records.")

        df = self._normalize_notional_aliases(df)
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

    def _normalize_orderbook_records(self, records: list[dict[str, Any]], detected: dict[str, Any]) -> dict[str, Any]:
        snapshots = pd.DataFrame([{key: value for key, value in row.items() if key not in {"bids", "asks"}} for row in records])
        if "timestamp" in snapshots.columns:
            snapshots["timestamp"] = pd.to_datetime(snapshots["timestamp"], utc=True, errors="coerce")
            snapshots = snapshots.sort_values("timestamp").reset_index(drop=True)

        last_snapshot = records[-1]
        bids = self._normalize_notional_aliases(pd.DataFrame(last_snapshot.get("bids", [])))
        asks = self._normalize_notional_aliases(pd.DataFrame(last_snapshot.get("asks", [])))
        if bids.empty or asks.empty:
            raise ValueError("Order book records require non-empty bids and asks in each snapshot.")

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
            "symbol": detected.get("symbol") or self._first_non_null(snapshots, "symbol"),
            "timeframe": detected.get("timeframe") or self._first_non_null(snapshots, "timeframe"),
            "dataframe": snapshots,
            "bids": bids,
            "asks": asks,
            "summary": {
                "kind": "orderbook_conventional",
                "rows": int(len(records)),
                "bid_levels": int(len(bids)),
                "ask_levels": int(len(asks)),
                "best_bid": best_bid,
                "best_ask": best_ask,
                "spread": best_ask - best_bid,
                "timestamp": last_snapshot.get("timestamp") or detected.get("timestamp_utc"),
            },
        }

    def _normalize_event_records(
        self,
        payload: dict[str, Any],
        detected: dict[str, Any],
        kind: str,
        event_type: str,
    ) -> dict[str, Any]:
        df = pd.DataFrame(payload.get("records", []))
        if df.empty:
            raise ValueError(f"{kind} payload has no records.")

        df = self._normalize_notional_aliases(df)
        df["event_type"] = event_type

        if "timestamp_utc" in df.columns:
            df = df.rename(columns={"timestamp_utc": "timestamp"})
        df = self._coalesce_duplicate_columns(df)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        if "placed_at" in df.columns:
            df["placed_at"] = pd.to_datetime(df["placed_at"], utc=True, errors="coerce")

        for col in ["price", "quantity_btc", "notional_usdt", "age_seconds", "active_duration_seconds", "active_duration_minutes"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        if "timestamp" in df.columns:
            df = df.sort_values("timestamp").reset_index(drop=True)

        return {
            "kind": kind,
            "symbol": detected.get("symbol") or self._first_non_null(df, "symbol"),
            "timeframe": detected.get("timeframe") or self._first_non_null(df, "timeframe"),
            "events": df,
            "summary": {
                "kind": kind,
                "rows": int(len(df)),
                "buy_rows": int((df["side"] == "buy").sum()) if "side" in df else 0,
                "sell_rows": int((df["side"] == "sell").sum()) if "side" in df else 0,
                "timestamp": detected.get("timestamp_utc"),
            },
        }

    def _normalize_records(self, payload: dict[str, Any], detected: dict[str, Any]) -> dict[str, Any]:
        records = payload.get("records", [])
        df = pd.DataFrame(records)
        if df.empty:
            raise ValueError(f"{detected['data_type']} payload has no records.")

        if "timestamp_utc" in df.columns:
            df = df.rename(columns={"timestamp_utc": "timestamp"})
        df = self._coalesce_duplicate_columns(df)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")

        for column in df.columns:
            if column in {"timestamp", "timeframe", "symbol", "aggressor_side", "flow_direction", "exchange_interpretation", "dominant_side_liquidated", "derivatives_pressure", "sentiment_regime"}:
                continue
            numeric = pd.to_numeric(df[column], errors="coerce")
            if not numeric.isna().all():
                df[column] = numeric

        if "timestamp" in df.columns:
            df = df.sort_values("timestamp").reset_index(drop=True)

        return {
            "kind": detected["data_type"],
            "symbol": detected.get("symbol") or self._first_non_null(df, "symbol"),
            "timeframe": detected.get("timeframe") or self._first_non_null(df, "timeframe"),
            "dataframe": df,
            "summary": {
                "kind": detected["data_type"],
                "rows": int(len(df)),
                "symbol": detected.get("symbol") or self._first_non_null(df, "symbol"),
                "timeframe": detected.get("timeframe") or self._first_non_null(df, "timeframe"),
                "start": df["timestamp"].min().isoformat() if "timestamp" in df.columns and not df["timestamp"].isna().all() else None,
                "end": df["timestamp"].max().isoformat() if "timestamp" in df.columns and not df["timestamp"].isna().all() else None,
                "columns": list(df.columns),
            },
        }

    def _normalize_metric_records(self, payload: dict[str, Any], detected: dict[str, Any]) -> dict[str, Any]:
        records = payload.get("records")
        if isinstance(records, list):
            rows = records
        else:
            rows = [{key: value for key, value in payload.items() if key != "metadata"}]

        df = pd.DataFrame(rows)
        if df.empty:
            raise ValueError(f"{detected['data_type']} payload has no metric records.")

        if "timestamp_utc" in df.columns:
            df = df.rename(columns={"timestamp_utc": "timestamp"})
        df = self._coalesce_duplicate_columns(df)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")

        for column in df.columns:
            if column in {"timestamp", "timeframe", "symbol"}:
                continue
            numeric = pd.to_numeric(df[column], errors="coerce")
            if not numeric.isna().all():
                df[column] = numeric

        if "timestamp" in df.columns:
            df = df.sort_values("timestamp").reset_index(drop=True)

        return {
            "kind": detected["data_type"],
            "symbol": detected.get("symbol") or self._first_non_null(df, "symbol"),
            "timeframe": detected.get("timeframe") or self._first_non_null(df, "timeframe"),
            "dataframe": df,
            "summary": {
                "kind": detected["data_type"],
                "rows": int(len(df)),
                "symbol": detected.get("symbol") or self._first_non_null(df, "symbol"),
                "timeframe": detected.get("timeframe") or self._first_non_null(df, "timeframe"),
                "start": df["timestamp"].min().isoformat() if "timestamp" in df.columns and not df["timestamp"].isna().all() else None,
                "end": df["timestamp"].max().isoformat() if "timestamp" in df.columns and not df["timestamp"].isna().all() else None,
                "columns": list(df.columns),
            },
        }

    @staticmethod
    def _first_non_null(df: pd.DataFrame, column: str) -> Any:
        if column not in df.columns:
            return None
        values = df[column].dropna()
        return values.iloc[0] if not values.empty else None

    @staticmethod
    def _coalesce_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
        if df.columns.is_unique:
            return df

        result = pd.DataFrame(index=df.index)
        for column in dict.fromkeys(df.columns):
            values = df.loc[:, df.columns == column]
            if isinstance(values, pd.Series):
                result[column] = values
            elif values.shape[1] == 1:
                result[column] = values.iloc[:, 0]
            else:
                result[column] = values.bfill(axis=1).iloc[:, 0]
        return result

    def _normalize_notional_aliases(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.rename(columns={"notional_usd": "notional_usdt"})
        return self._coalesce_duplicate_columns(df)
