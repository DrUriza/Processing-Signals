import numpy as np
import pandas as pd
import pytest

from signal_analysis import (
    detectar_compra,
    detectar_venta,
    high_volume,
    stop_loss_take_profit,
)


def test_detectar_compra_marks_only_activation_edges():
    signal = [0.0, 1.0, 1.0, 0.0, 1.0]
    out = detectar_compra(signal)
    assert isinstance(out, np.ndarray)
    assert out.tolist() == [0.0, 1.0, 0.0, 0.0, 1.0]


def test_detectar_venta_marks_negative_activation_edges():
    signal = [0.0, 1.0, 1.0, 0.0, 1.0]
    out = detectar_venta(signal)
    assert isinstance(out, np.ndarray)
    assert out.tolist() == [0.0, -1.0, 0.0, 0.0, -1.0]


def test_high_volume_returns_binary_flag():
    df = pd.DataFrame({"volume": [100] * 500 + [300, 310, 320, 330, 340]})
    out = high_volume(df, lookback=500, n_last=5, k=1.0)
    assert out in (0, 1)
    assert out == 1


def test_stop_loss_take_profit_bullish_case_returns_ordered_levels():
    data = pd.DataFrame(
        {
            "close": [100.0],
            "ATR": [2.0],
            "ADX": [30.0],
            "MD+": [35.0],
            "MD-": [20.0],
        }
    )
    sl, tp = stop_loss_take_profit(data)
    assert sl < data["close"].iloc[-1] < tp


def test_stop_loss_take_profit_bearish_case_returns_ordered_levels():
    data = pd.DataFrame(
        {
            "close": [100.0],
            "ATR": [2.0],
            "ADX": [30.0],
            "MD+": [10.0],
            "MD-": [25.0],
        }
    )
    sl, tp = stop_loss_take_profit(data)
    assert tp < data["close"].iloc[-1] < sl


def test_stop_loss_take_profit_invalid_atr_raises_value_error():
    data = pd.DataFrame(
        {
            "close": [100.0],
            "ATR": [0.0],
            "ADX": [30.0],
            "MD+": [10.0],
            "MD-": [25.0],
        }
    )
    with pytest.raises(ValueError, match="ATR"):
        stop_loss_take_profit(data)
