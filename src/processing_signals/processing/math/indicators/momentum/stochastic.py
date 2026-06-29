from __future__ import annotations

import pandas as pd


def stochastic(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14, smooth: int = 3) -> pd.DataFrame:
    high = pd.to_numeric(high, errors="coerce")
    low = pd.to_numeric(low, errors="coerce")
    close = pd.to_numeric(close, errors="coerce")
    lowest = low.rolling(window).min()
    highest = high.rolling(window).max()
    k = 100 * (close - lowest) / (highest - lowest).replace(0, pd.NA)
    d = k.rolling(smooth).mean()
    return pd.DataFrame({"stoch_k_14": k, "stoch_d_14": d})
