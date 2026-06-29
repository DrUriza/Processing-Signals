from __future__ import annotations

import pandas as pd


def momentum(close: pd.Series, window: int = 10) -> pd.Series:
    return pd.to_numeric(close, errors="coerce").diff(window)
