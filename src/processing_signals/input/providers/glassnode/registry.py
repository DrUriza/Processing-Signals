from __future__ import annotations

GLASSNODE_ENDPOINTS = {
    "mvrv": {"path": "/v1/metrics/market/mvrv", "family": "market_regime", "data_type": "onchain_holder_behavior"},
    "sopr": {"path": "/v1/metrics/indicators/sopr", "family": "market_regime", "data_type": "onchain_holder_behavior"},
    "realized_price": {"path": "/v1/metrics/market/price_realized_usd", "family": "market_regime", "data_type": "onchain_holder_behavior"},
    "realized_cap": {"path": "/v1/metrics/market/realized_cap", "family": "market_regime", "data_type": "onchain_holder_behavior"},
    "supply_liquidity": {"path": "/v1/metrics/supply/liquid_illiquid_sum", "family": "market_regime", "data_type": "onchain_holder_behavior"},
    "institutional_flows": {"path": "/v1/metrics/institutions/etf_flows_sum", "family": "exchange_flows", "data_type": "etf_and_exchange_flows"},
    "options_volatility": {"path": "/v1/metrics/options/atm_implied_volatility_1_week", "family": "market_regime", "data_type": "options_volatility"},
    "miner_metrics": {"path": "/v1/metrics/mining/hash_rate_mean", "family": "on_chain_miners", "data_type": "mining_network_health"},
}
