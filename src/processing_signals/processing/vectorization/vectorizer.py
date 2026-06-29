from __future__ import annotations

from typing import Any


class Vectorizer:
    """Build base vectorization metadata before transform/math layers."""

    EXCLUDED_COLUMNS = {
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
    }

    def vectorize(self, normalized: dict[str, Any], detected: dict[str, Any]) -> dict[str, Any]:
        dataframe = normalized.get("dataframe")
        events = normalized.get("events")
        numeric_columns: list[str] = []
        if hasattr(dataframe, "select_dtypes"):
            numeric_columns = self._numeric_columns(dataframe)
        elif hasattr(events, "select_dtypes"):
            numeric_columns = self._numeric_columns(events)

        return {
            "source_name": detected.get("source_name"),
            "data_type": detected.get("data_type"),
            "symbol": detected.get("symbol"),
            "timeframe": detected.get("timeframe"),
            "numeric_columns": numeric_columns,
        }

    def _numeric_columns(self, frame: Any) -> list[str]:
        return [
            str(column)
            for column in frame.select_dtypes(include="number").columns
            if str(column).lower() not in self.EXCLUDED_COLUMNS
        ]
