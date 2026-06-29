from __future__ import annotations

CRYPTOQUANT_ENDPOINTS = {
    "exchange_reserve": {"path": "/v1/btc/exchange-flows/reserve", "family": "exchange_flows", "data_type": "exchange_reserve"},
    "exchange_netflow": {"path": "/v1/btc/exchange-flows/netflow", "family": "exchange_flows", "data_type": "exchange_netflow"},
    "exchange_inflow": {"path": "/v1/btc/exchange-flows/inflow", "family": "exchange_flows", "data_type": "exchange_inflow"},
    "exchange_outflow": {"path": "/v1/btc/exchange-flows/outflow", "family": "exchange_flows", "data_type": "exchange_outflow"},
    "miner_flows": {"path": "/v1/btc/miner-flows", "family": "on_chain_miners", "data_type": "mining_network_health"},
    "market_indicators": {"path": "/v1/btc/market-indicators", "family": "market_regime", "data_type": "market_indicators"},
}
