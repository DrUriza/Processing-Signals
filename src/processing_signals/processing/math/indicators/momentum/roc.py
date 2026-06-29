from __future__ import annotations

import pandas as pd


def roc(close: pd.Series, window: int = 12) -> pd.Series:
    close = pd.to_numeric(close, errors="coerce")
    return close.pct_change(window) * 100
