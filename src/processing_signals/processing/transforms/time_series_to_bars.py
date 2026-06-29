from __future__ import annotations

from typing import Any

import pandas as pd


def reference_column(df: pd.DataFrame, preferred: str | None = None) -> str | None:
    if preferred and preferred in df.columns:
        return preferred
    for candidate in ["close", "value", "open_interest_usd", "long_short_ratio", "exchange_netflow", "total_liquidations_usd"]:
        if candidate in df.columns:
            return candidate
    numeric_df = df.select_dtypes(include="number")
    if not numeric_df.empty:
        return str(numeric_df.columns[0])
    return None


def time_series_to_bars(df: pd.DataFrame, preferred_reference: str | None = None) -> tuple[list[dict[str, Any]], str | None]:
    """Create OHLC-compatible bar records from OHLC data or one numeric series."""
    if df.empty:
        return [], None

    ref = reference_column(df, preferred_reference)
    has_ohlc = all(column in df.columns for column in ["open", "high", "low", "close"])
    bars: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        record: dict[str, Any] = {"timestamp": row.get("timestamp")}
        if has_ohlc:
            record.update({key: row.get(key) for key in ["open", "high", "low", "close"]})
        elif ref:
            value = row.get(ref)
            record.update({"open": value, "high": value, "low": value, "close": value})
        else:
            continue
        if "volume" in df.columns:
            record["volume"] = row.get("volume")
        elif "notional_volume" in df.columns:
            record["volume"] = row.get("notional_volume")
        else:
            record["volume"] = None
        bars.append(record)
    return bars, ref
