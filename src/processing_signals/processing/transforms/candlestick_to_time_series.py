from __future__ import annotations

import pandas as pd


def candlestick_to_time_series(df: pd.DataFrame) -> pd.DataFrame:
    """Return OHLCV records in time-series compatible shape."""
    columns = [column for column in ["timestamp", "symbol", "timeframe", "open", "high", "low", "close", "volume", "notional_volume"] if column in df.columns]
    return df.loc[:, columns].copy()
