from __future__ import annotations

import pandas as pd


def donchian(high: pd.Series, low: pd.Series, window: int = 20) -> pd.DataFrame:
    upper = pd.to_numeric(high, errors="coerce").rolling(window).max()
    lower = pd.to_numeric(low, errors="coerce").rolling(window).min()
    return pd.DataFrame({"donchian_upper_20": upper, "donchian_lower_20": lower, "donchian_mid_20": (upper + lower) / 2})
