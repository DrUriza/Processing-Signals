from __future__ import annotations

import numpy as np
import pandas as pd


def _min_periods(window: int, min_periods: int | None) -> int:
    return window if min_periods is None else min_periods


def rolling_drawdown(series: pd.Series, window: int, min_periods: int | None = None) -> pd.Series:
    """Rolling drawdown relative to the rolling maximum."""
    s = pd.to_numeric(series.copy(), errors="coerce")
    rolling_peak = s.rolling(window, min_periods=_min_periods(window, min_periods)).max().replace(0, np.nan)
    return (s / rolling_peak) - 1


def rolling_max_drawdown(series: pd.Series, window: int, min_periods: int | None = None) -> pd.Series:
    """Worst drawdown observed inside each rolling window."""
    drawdown = rolling_drawdown(series, window, min_periods)
    return drawdown.rolling(window, min_periods=_min_periods(window, min_periods)).min()
