"""Endpoint registry for useful CryptoQuant API endpoints."""

from processing_signals.input.apis.registry_helpers import endpoint


PROVIDER = "cryptoquant"
BASE_URL = "https://api.cryptoquant.com/v1"
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
        notes=notes or "Useful CryptoQuant metric; live path is not configured yet.",
    )


ENDPOINTS = [
    # institutional_flows: 5
    _metric("institutional_flows", "exchange_inflow", "primary", notes="CryptoQuant is primary for on-chain exchange inflow."),
    _metric("institutional_flows", "exchange_outflow", "primary", notes="CryptoQuant is primary for on-chain exchange outflow."),
    _metric("institutional_flows", "exchange_netflow", "primary", notes="CryptoQuant is primary for on-chain exchange netflow."),
    _metric("institutional_flows", "exchange_reserve", "primary", notes="CryptoQuant is primary for on-chain exchange reserve."),
    _metric("institutional_flows", "stablecoin_flows", "secondary", notes="CryptoQuant is useful for stablecoin flow context."),

    # derivatives_open_interest: 3
    _metric("derivatives_open_interest", "open_interest", "tertiary", notes="CryptoQuant is structural/fallback context for open interest."),
    _metric("derivatives_open_interest", "funding_rate", "tertiary", notes="CryptoQuant is structural/fallback context for funding rate."),
    _metric("derivatives_open_interest", "estimated_leverage_ratio", "primary", notes="CryptoQuant is primary for estimated leverage ratio."),

    # sentiment_positioning: 2
    _metric("sentiment_positioning", "long_short_ratio", "secondary", notes="CryptoQuant provides derived positioning context."),
    _metric("sentiment_positioning", "sentiment_index", "secondary", notes="CryptoQuant provides derived sentiment context."),

    # on_chain_miners: 7
    _metric("on_chain_miners", "miner_reserve", "primary", notes="CryptoQuant is primary for miner reserve."),
    _metric("on_chain_miners", "miner_inflow", "primary", notes="CryptoQuant is primary for miner inflow."),
    _metric("on_chain_miners", "miner_outflow", "primary", notes="CryptoQuant is primary for miner outflow."),
    _metric("on_chain_miners", "miner_outflow_multiple", "primary", notes="CryptoQuant is primary for miner outflow multiple."),
    _metric("on_chain_miners", "mpi", "primary", notes="CryptoQuant is primary for Miners' Position Index."),
    _metric("on_chain_miners", "hash_rate", "secondary", notes="CryptoQuant is useful for hash rate context."),
    _metric("on_chain_miners", "difficulty", "secondary", notes="CryptoQuant is useful for mining difficulty context."),
]
