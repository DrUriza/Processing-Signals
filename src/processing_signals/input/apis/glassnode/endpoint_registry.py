"""Endpoint registry for useful Glassnode API endpoints."""

from processing_signals.input.apis.registry_helpers import endpoint


PROVIDER = "glassnode"
BASE_URL = "https://api.glassnode.com/v1"
SYNTHETIC_TIMEFRAMES = ["1m", "5m", "15m", "1h"]


def _metric(family: str, subtype: str, role: str, status: str = "partial", notes: str = "") -> dict:
    return endpoint(
        provider=PROVIDER,
        family=family,
        subtype=subtype,
        path=None,
        coverage_role=role,
        live_status=status,
        data_type="time_series",
        supports_timeframe=True,
        synthetic_timeframes=SYNTHETIC_TIMEFRAMES,
        default_params={},
        notes=notes or "Useful Glassnode metric; live path is not configured yet.",
    )


def _candles(family: str, subtype: str, role: str, status: str = "partial", notes: str = "") -> dict:
    item = _metric(family, subtype, role, status, notes)
    item["data_type"] = "candlestick"
    return item


def _snapshot(family: str, subtype: str, role: str, notes: str = "") -> dict:
    return endpoint(
        provider=PROVIDER,
        family=family,
        subtype=subtype,
        path=None,
        coverage_role=role,
        live_status="partial",
        data_type="snapshot",
        supports_timeframe=False,
        extraction_windows=["latest"],
        default_params={},
        notes=notes or "Useful Glassnode snapshot metric; live path is not configured yet.",
    )


def _heatmap(family: str, subtype: str, role: str, notes: str = "") -> dict:
    return endpoint(
        provider=PROVIDER,
        family=family,
        subtype=subtype,
        path=None,
        coverage_role=role,
        live_status="partial",
        data_type="heatmap",
        supports_timeframe=False,
        extraction_windows=["24h"],
        default_params={},
        notes=notes or "Useful Glassnode heatmap metric; live path is not configured yet.",
    )


ENDPOINTS = [
    # prices_ohlcv: 3
    _candles("prices_ohlcv", "spot_ohlcv", "secondary", notes="Glassnode provides market price context."),
    _candles("prices_ohlcv", "futures_ohlcv", "secondary", notes="Derived futures OHLCV context from Glassnode market metrics."),
    _metric("prices_ohlcv", "index_price", "secondary", notes="Glassnode provides index/market price context."),

    # volume_orderflow: 3
    _metric("volume_orderflow", "cvd", "secondary", notes="Derived aggregate context, not 1m operational order flow."),
    _metric("volume_orderflow", "volume_delta", "secondary", notes="Derived aggregate context, not 1m operational order flow."),
    _metric("volume_orderflow", "buy_sell_pressure", "secondary", notes="Derived aggregate context, not 1m operational order flow."),

    # institutional_flows: 6
    _metric("institutional_flows", "etf_flows", "primary", notes="Glassnode is useful for institutions and ETF metrics."),
    _metric("institutional_flows", "exchange_inflow", "secondary", notes="Glassnode is useful for on-chain exchange inflow."),
    _metric("institutional_flows", "exchange_outflow", "secondary", notes="Glassnode is useful for on-chain exchange outflow."),
    _metric("institutional_flows", "exchange_netflow", "secondary", notes="Glassnode is useful for on-chain exchange netflow."),
    _metric("institutional_flows", "exchange_reserve", "secondary", notes="Glassnode is useful for exchange reserve."),
    _metric("institutional_flows", "stablecoin_flows", "primary", notes="Glassnode is useful for stablecoin flow context."),

    # liquidations: 3
    _metric("liquidations", "long_liquidations", "secondary", notes="Glassnode provides structural liquidation context."),
    _metric("liquidations", "short_liquidations", "secondary", notes="Glassnode provides structural liquidation context."),
    _metric("liquidations", "long_short_liquidations", "secondary", notes="Glassnode provides structural liquidation context."),

    # derivatives_open_interest: 6
    _metric("derivatives_open_interest", "open_interest", "secondary", notes="Glassnode is secondary/fallback for open interest."),
    _metric("derivatives_open_interest", "open_interest_change", "secondary", notes="Derived from Glassnode open interest context."),
    _metric("derivatives_open_interest", "funding_rate", "secondary", notes="Glassnode is secondary/fallback for funding rate."),
    _metric("derivatives_open_interest", "basis", "secondary", notes="Glassnode is secondary/fallback for basis."),
    _metric("derivatives_open_interest", "futures_premium", "secondary", notes="Glassnode is secondary/fallback for futures premium."),
    _metric("derivatives_open_interest", "estimated_leverage_ratio", "secondary", notes="Glassnode is secondary/fallback for ELR."),

    # sentiment_positioning: 3
    _metric("sentiment_positioning", "long_short_ratio", "tertiary", notes="Glassnode provides derived positioning context."),
    _metric("sentiment_positioning", "fear_greed", "tertiary", notes="Derived sentiment context."),
    _metric("sentiment_positioning", "sentiment_index", "tertiary", notes="Derived sentiment index context."),

    # on_chain_miners: 8
    _metric("on_chain_miners", "miner_reserve", "secondary", notes="Glassnode mining category coverage."),
    _metric("on_chain_miners", "miner_inflow", "secondary", notes="Glassnode mining category coverage."),
    _metric("on_chain_miners", "miner_outflow", "secondary", notes="Glassnode mining category coverage."),
    _metric("on_chain_miners", "miner_outflow_multiple", "secondary", notes="Glassnode mining category coverage."),
    _metric("on_chain_miners", "mpi", "secondary", notes="Glassnode mining category coverage."),
    _metric("on_chain_miners", "hash_rate", "primary", notes="Glassnode mining category coverage."),
    _metric("on_chain_miners", "difficulty", "primary", notes="Glassnode mining category coverage."),
    _metric("on_chain_miners", "miner_revenue", "primary", notes="Glassnode mining category coverage."),

    # options_volatility: 8
    _metric("options_volatility", "implied_volatility", "primary", notes="Glassnode options category coverage."),
    _metric("options_volatility", "realized_volatility", "primary", notes="Glassnode options category coverage."),
    _metric("options_volatility", "options_open_interest", "primary", notes="Glassnode options category coverage."),
    _metric("options_volatility", "options_volume", "primary", notes="Glassnode options category coverage."),
    _metric("options_volatility", "put_call_ratio", "primary", notes="Derived from Glassnode options OI/volume context."),
    _metric("options_volatility", "skew", "primary", notes="Glassnode options category coverage."),
    _heatmap("options_volatility", "gamma_exposure", "primary", notes="Glassnode options distribution/GEX coverage."),
    _snapshot("options_volatility", "max_pain", "primary", notes="Glassnode options max pain coverage."),
]
