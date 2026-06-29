from __future__ import annotations

from typing import Any

import pandas as pd

from processing_signals.processing.transforms.candlestick_to_time_series import candlestick_to_time_series
from processing_signals.processing.transforms.event_extractor import extract_events
from processing_signals.processing.transforms.orderbook_to_bars import orderbook_to_bars
from processing_signals.processing.transforms.time_series_to_bars import time_series_to_bars
from processing_signals.processing.transforms.time_series_to_candlestick import time_series_to_candlestick


class TransformEngine:
    """Prepare reusable view transforms before math/output layers."""

    def transform(
        self,
        normalized: dict[str, Any],
        detected: dict[str, Any],
        vectorized: dict[str, Any],
    ) -> dict[str, Any]:
        views: dict[str, Any] = {}
        dataframe = normalized.get("dataframe")
        if isinstance(dataframe, pd.DataFrame):
            preferred = vectorized.get("numeric_columns", [None])[0] if vectorized.get("numeric_columns") else None
            bars, reference = orderbook_to_bars(dataframe) if normalized.get("kind") == "orderbook_conventional" else time_series_to_bars(dataframe, preferred)
            candles, conversion = time_series_to_candlestick(dataframe, reference)
            views["bars"] = {"records": bars, "reference_column": reference}
            views["candlestick_derived"] = {"records": candles, "conversion": conversion}
            views["time_series"] = candlestick_to_time_series(dataframe) if normalized.get("kind") == "candlestick" else dataframe
            views["event_list"] = {
                "events": extract_events(
                    dataframe,
                    vectorized.get("numeric_columns", []),
                    detected.get("metadata", {}).get("family") or "",
                    detected.get("data_type"),
                    detected.get("source_name"),
                    detected.get("symbol"),
                    detected.get("timeframe"),
                )
            }

        events = normalized.get("events")
        if isinstance(events, pd.DataFrame):
            views["event_list"] = {"events": events}
        if normalized.get("kind") == "orderbook_conventional":
            views["orderbook"] = {
                "bids": normalized.get("bids"),
                "asks": normalized.get("asks"),
                "summary": normalized.get("summary", {}),
            }
        return views
