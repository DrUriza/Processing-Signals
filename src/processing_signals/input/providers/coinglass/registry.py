from __future__ import annotations

COINGLASS_ENDPOINTS = {
    "futures_price_ohlc": {
        "path": "/api/futures/price/history",
        "family": "prices_ohlcv",
        "data_type": "candlestick",
    },
    "open_interest_history": {
        "path": "/api/futures/open-interest/history",
        "family": "derivatives_open_interest",
        "data_type": "open_interest_and_funding",
    },
    "funding_rate_history": {
        "path": "/api/futures/funding-rate/history",
        "family": "derivatives_open_interest",
        "data_type": "open_interest_and_funding",
    },
    "liquidation_history": {
        "path": "/api/futures/liquidation/history",
        "family": "liquidations",
        "data_type": "long_short_liquidations",
    },
    "long_short_ratio": {
        "path": "/api/futures/long-short-ratio/history",
        "family": "market_regime",
        "data_type": "long_short_ratio",
    },
    "etf_flow_history": {
        "path": "/api/etf/flow/history",
        "family": "exchange_flows",
        "data_type": "etf_and_exchange_flows",
    },
}
