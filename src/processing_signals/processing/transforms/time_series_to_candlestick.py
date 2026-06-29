from __future__ import annotations

from typing import Any

import pandas as pd

from processing_signals.processing.transforms.time_series_to_bars import time_series_to_bars


def time_series_to_candlestick(df: pd.DataFrame, preferred_reference: str | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Create derived candles from a numeric time-series reference column."""
    candles, reference = time_series_to_bars(df, preferred_reference)
    return candles, {
        "method": "numeric_series_to_ohlc",
        "reference_column": reference,
        "source_rows": int(len(df)),
        "derived_rows": len(candles),
    }
