"""Endpoint registry for useful CoinGlass API endpoints."""

from processing_signals.input.apis.registry_helpers import endpoint


PROVIDER = "coinglass"
BASE_URL = "https://open-api-v4.coinglass.com"
SYNTHETIC_TIMEFRAMES = ["1m", "5m", "15m", "1h"]
LIVE_PRICE_TIMEFRAMES = ["1m", "3m", "5m", "15m", "30m", "1h", "4h", "6h", "8h", "12h", "1d", "1w"]

_FUTURES_PARAMS = {"exchange": "Binance", "symbol": "BTCUSDT", "interval": "{timeframe}", "limit": 600}
_AGG_PARAMS = {"exchange_list": "Binance,OKX,Bybit", "symbol": "BTC", "interval": "{timeframe}", "limit": 600}
_OPTIONS_PARAMS = {"exchange": "Deribit", "symbol": "BTC", "interval": "{timeframe}", "limit": 600}


def _tf(family: str, subtype: str, path: str | None, role: str, status: str, params: dict | None = None) -> dict:
    return endpoint(
        provider=PROVIDER,
        family=family,
        subtype=subtype,
        path=path,
        coverage_role=role,
        live_status=status,
        data_type="time_series",
        supports_timeframe=True,
        live_supported_timeframes=LIVE_PRICE_TIMEFRAMES if path else [],
        synthetic_timeframes=SYNTHETIC_TIMEFRAMES,
        default_params=params or {},
        notes="Useful CoinGlass endpoint for this provider registry.",
    )


def _candles(family: str, subtype: str, path: str, status: str, params: dict) -> dict:
    item = _tf(family, subtype, path, "primary", status, params)
    item["data_type"] = "candlestick"
    return item


def _window(
    family: str,
    subtype: str,
    path: str,
    role: str,
    status: str,
    data_type: str,
    window: str,
    params: dict | None = None,
) -> dict:
    return endpoint(
        provider=PROVIDER,
        family=family,
        subtype=subtype,
        path=path,
        coverage_role=role,
        live_status=status,
        data_type=data_type,
        supports_timeframe=False,
        extraction_windows=[window],
        default_params=params or {},
        notes="Useful CoinGlass endpoint for this provider registry.",
    )


ENDPOINTS = [
    # prices_ohlcv: 4
    _candles("prices_ohlcv", "spot_ohlcv", "/api/spot/price/history", "supported", _FUTURES_PARAMS),
    _candles("prices_ohlcv", "futures_ohlcv", "/api/futures/price/history", "supported", _FUTURES_PARAMS),
    _tf("prices_ohlcv", "index_price", "/api/spot/price/history", "primary", "derived", _FUTURES_PARAMS),
    _tf("prices_ohlcv", "mark_price", "/api/futures/price/history", "primary", "derived", _FUTURES_PARAMS),

    # liquidity_microstructure: 6
    _tf("liquidity_microstructure", "orderbook_conventional", "/api/futures/orderbook/ask-bids-history", "primary", "supported", _FUTURES_PARAMS),
    _window("liquidity_microstructure", "orderbook_large_trades", "/api/futures/orderbook/large-limit-order", "primary", "supported", "event_list", "latest", {"exchange": "Binance", "symbol": "BTCUSDT"}),
    _window("liquidity_microstructure", "orderbook_whale_orders", "/api/futures/orderbook/large-limit-order", "primary", "derived", "event_list", "latest", {"exchange": "Binance", "symbol": "BTCUSDT"}),
    _tf("liquidity_microstructure", "market_depth", "/api/futures/orderbook/aggregated-ask-bids-history", "primary", "supported", _AGG_PARAMS),
    _tf("liquidity_microstructure", "bid_ask_spread", "/api/futures/orderbook/ask-bids-history", "primary", "derived", _FUTURES_PARAMS),
    _window("liquidity_microstructure", "liquidity_walls", "/api/futures/orderbook/large-limit-order", "primary", "derived", "event_list", "latest", {"exchange": "Binance", "symbol": "BTCUSDT"}),

    # volume_orderflow: 5
    _tf("volume_orderflow", "cvd", "/api/futures/cvd/history", "primary", "supported", _FUTURES_PARAMS),
    _tf("volume_orderflow", "taker_buy_sell_volume", "/api/futures/aggregated-taker-buy-sell-volume/history", "primary", "supported", _AGG_PARAMS),
    _tf("volume_orderflow", "volume_delta", "/api/futures/cvd/history", "primary", "derived", _FUTURES_PARAMS),
    _tf("volume_orderflow", "buy_sell_pressure", "/api/futures/aggregated-taker-buy-sell-volume/history", "primary", "derived", _AGG_PARAMS),
    _tf("volume_orderflow", "aggressive_volume", "/api/futures/volume/footprint-history", "primary", "derived", _FUTURES_PARAMS),

    # institutional_flows: 2
    _tf("institutional_flows", "etf_flows", "/api/etf/bitcoin/flow-history", "tertiary", "supported", {"symbol": "BTC", "interval": "{timeframe}", "limit": 600}),
    _tf("institutional_flows", "stablecoin_flows", "/api/index/stableCoin-marketCap-history", "tertiary", "partial", {"interval": "{timeframe}", "limit": 600}),

    # liquidations: 6
    _tf("liquidations", "long_liquidations", "/api/futures/liquidation/aggregated-history", "primary", "supported", _AGG_PARAMS),
    _tf("liquidations", "short_liquidations", "/api/futures/liquidation/aggregated-history", "primary", "supported", _AGG_PARAMS),
    _tf("liquidations", "long_short_liquidations", "/api/futures/liquidation/aggregated-history", "primary", "supported", _AGG_PARAMS),
    _window("liquidations", "liquidation_heatmap", "/api/futures/liquidation/aggregated-heatmap/model1", "primary", "supported", "heatmap", "24h", {"exchange_list": "Binance,OKX,Bybit", "symbol": "BTC"}),
    _window("liquidations", "liquidation_clusters", "/api/futures/liquidation/aggregated-heatmap/model1", "primary", "derived", "event_list", "24h", {"exchange_list": "Binance,OKX,Bybit", "symbol": "BTC"}),
    _window("liquidations", "liquidation_map", "/api/futures/liquidation/aggregated-map", "primary", "supported", "snapshot", "latest", {"exchange_list": "Binance,OKX,Bybit", "symbol": "BTC"}),

    # derivatives_open_interest: 5
    _tf("derivatives_open_interest", "open_interest", "/api/futures/open-interest/history", "primary", "supported", _FUTURES_PARAMS),
    _tf("derivatives_open_interest", "open_interest_change", "/api/futures/open-interest/history", "primary", "derived", _FUTURES_PARAMS),
    _tf("derivatives_open_interest", "funding_rate", "/api/futures/funding-rate/history", "primary", "supported", _FUTURES_PARAMS),
    _tf("derivatives_open_interest", "basis", "/api/futures/basis/history", "primary", "supported", _FUTURES_PARAMS),
    _tf("derivatives_open_interest", "futures_premium", "/api/futures/basis/history", "primary", "derived", _FUTURES_PARAMS),

    # sentiment_positioning: 6
    _tf("sentiment_positioning", "long_short_ratio", "/api/futures/global-long-short-account-ratio/history", "primary", "supported", _FUTURES_PARAMS),
    _tf("sentiment_positioning", "top_trader_long_short_ratio", "/api/futures/top-long-short-account-ratio/history", "primary", "supported", _FUTURES_PARAMS),
    _tf("sentiment_positioning", "account_long_short_ratio", "/api/futures/global-long-short-account-ratio/history", "primary", "supported", _FUTURES_PARAMS),
    _tf("sentiment_positioning", "position_long_short_ratio", "/api/futures/top-long-short-position-ratio/history", "primary", "supported", _FUTURES_PARAMS),
    _tf("sentiment_positioning", "fear_greed", "/api/index/fear-greed-history", "primary", "supported", {"interval": "{timeframe}", "limit": 600}),
    _tf("sentiment_positioning", "sentiment_index", "/api/index/fear-greed-history", "primary", "derived", {"interval": "{timeframe}", "limit": 600}),

    # options_volatility: 4
    _tf("options_volatility", "options_open_interest", "/api/option/exchange-oi-history", "secondary", "supported", _OPTIONS_PARAMS),
    _tf("options_volatility", "options_volume", "/api/option/exchange-vol-history", "secondary", "supported", _OPTIONS_PARAMS),
    _tf("options_volatility", "put_call_ratio", "/api/option/exchange-oi-history", "secondary", "derived", _OPTIONS_PARAMS),
    _window("options_volatility", "max_pain", "/api/option/max-pain", "secondary", "supported", "snapshot", "latest", {"exchange": "Deribit", "symbol": "BTC"}),
]
