from __future__ import annotations

import numpy as np
import pandas as pd


def ema(series: pd.Series, span: int) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").ewm(span=span, adjust=False).mean()


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    close = pd.to_numeric(close, errors="coerce")
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    close = pd.to_numeric(close, errors="coerce")
    macd_line = ema(close, fast) - ema(close, slow)
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return pd.DataFrame(
        {
            "macd": macd_line,
            "macd_signal": signal_line,
            "macd_hist": hist,
        }
    )


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    high = pd.to_numeric(high, errors="coerce")
    low = pd.to_numeric(low, errors="coerce")
    close = pd.to_numeric(close, errors="coerce")
    prev_close = close.shift(1)
    return pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    tr = true_range(high, low, close)
    return tr.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()


def bollinger_bands(close: pd.Series, window: int = 20, std_mult: float = 2.0) -> pd.DataFrame:
    close = pd.to_numeric(close, errors="coerce")
    middle = close.rolling(window).mean()
    std = close.rolling(window).std()
    upper = middle + std_mult * std
    lower = middle - std_mult * std
    bandwidth = (upper - lower) / middle.replace(0, np.nan)
    percent_b = (close - lower) / (upper - lower).replace(0, np.nan)
    return pd.DataFrame(
        {
            f"bb_middle_{window}": middle,
            f"bb_upper_{window}": upper,
            f"bb_lower_{window}": lower,
            f"bb_bandwidth_{window}": bandwidth,
            f"bb_percent_b_{window}": percent_b,
        }
    )


def vwap(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
    typical_price = (pd.to_numeric(high, errors="coerce") + pd.to_numeric(low, errors="coerce") + pd.to_numeric(close, errors="coerce")) / 3
    volume = pd.to_numeric(volume, errors="coerce")
    return (typical_price * volume).cumsum() / volume.cumsum().replace(0, np.nan)


def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    close = pd.to_numeric(close, errors="coerce")
    volume = pd.to_numeric(volume, errors="coerce").fillna(0)
    direction = np.sign(close.diff()).fillna(0)
    return (direction * volume).cumsum()


def compute_ohlcv_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["rsi_14"] = rsi(df["close"], 14)
    out = pd.concat([out, macd(df["close"])], axis=1)
    out["atr_14"] = atr(df["high"], df["low"], df["close"], 14)
    out = pd.concat([out, bollinger_bands(df["close"], 20, 2.0)], axis=1)
    out["ema_20"] = ema(df["close"], 20)
    out["ema_50"] = ema(df["close"], 50)
    out["vwap"] = vwap(df["high"], df["low"], df["close"], df["volume"])
    out["obv"] = obv(df["close"], df["volume"])
    return out
