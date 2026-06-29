from __future__ import annotations

import pandas as pd


def fibonacci_levels(high: pd.Series, low: pd.Series, window: int = 100) -> pd.DataFrame:
    high_roll = pd.to_numeric(high, errors="coerce").rolling(window).max()
    low_roll = pd.to_numeric(low, errors="coerce").rolling(window).min()
    diff = high_roll - low_roll
    return pd.DataFrame({
        "fib_236_100": high_roll - diff * 0.236,
        "fib_382_100": high_roll - diff * 0.382,
        "fib_500_100": high_roll - diff * 0.500,
        "fib_618_100": high_roll - diff * 0.618,
    })
