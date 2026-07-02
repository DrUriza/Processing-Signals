from __future__ import annotations
from typing     import Any


def resolve_output_family(block: dict[str, Any]) -> dict[str, str]:
    detected = block.get("detected", {})
    normalized = block.get("normalized", {})
    data_type = str(detected.get("data_type") or "").lower()
    canonical_type = str(detected.get("canonical_type") or "").lower()
    kind = str(normalized.get("kind") or "").lower()
    shape_source = " ".join([data_type, canonical_type, kind])

    if any(
        token in data_type
        for token in ["miner", "mining", "hash_rate", "hashrate", "miner_ratio","miner_inflow",
                      "miner_outflow", "miner_netflow", "difficulty", "network_health"]):
        output_shape = _network_or_onchain_shape(shape_source)
        return _family("mining_network_health", output_shape, f"{output_shape}.json")

    if any(
        token in data_type
        for token in [
            "onchain",
            "on_chain",
            "glassnode",
            "holder",
            "holders",
            "cohort",
            "cohorts",
            "accumulation",
            "distribution",
            "exchange_balance",
            "mvrv",
            "nvt",
            "sopr",
            "realized_cap",
            "supply_in_profit",
            "supply_in_loss",
            "long_term_holder",
            "short_term_holder",
        ]
    ):
        output_shape = _network_or_onchain_shape(shape_source)
        return _family("onchain_holder_behavior", output_shape, f"{output_shape}.json")

    if data_type == "candlestick":
        return _family("prices_ohlcv", "candlestick", "candlestick.json")

    if data_type == "orderbook_conventional":
        return _family("liquidity_microstructure", "conventional_orderbook", "conventional_orderbook.json")

    if data_type == "orderbook_large_trades":
        return _family("liquidity_microstructure", "large_trades_orderbook", "large_trades_orderbook.json")

    if data_type == "orderbook_whale_orders":
        return _family(
            "liquidity_microstructure",
            "whale_orders_orderbook",
            "whale_orders_orderbook.json",
        )

    if data_type == "manifest":
        return _family("metadata", "manifest", "manifest.json", is_metadata=True)

    if "cvd" in data_type or "volume" in data_type:
        output_shape = _volume_orderflow_shape(data_type, shape_source)
        return _family("volume_orderflow", output_shape, f"{output_shape}.json")

    if any(token in data_type for token in ["etf", "exchange_flow", "netflow", "inflow", "outflow"]):
        output_shape = _institutional_shape(shape_source)
        return _family("institutional_flows", output_shape, f"{output_shape}.json")

    if "liquidation" in data_type:
        output_shape = _institutional_shape(shape_source)
        return _family("liquidations", output_shape, f"{output_shape}.json")

    if "long_short" in data_type:
        output_shape = _sentiment_shape(shape_source)
        return _family("sentiment_positioning", output_shape, f"{output_shape}.json")

    if "open_interest" in data_type or data_type == "oi":
        output_shape = _open_interest_shape(shape_source)
        return _family("derivatives_open_interest", output_shape, f"{output_shape}.json")

    if "whale" in data_type:
        return _family(
            "liquidity_microstructure",
            "whale_orders_orderbook",
            "whale_orders_orderbook.json",
        )

    return _family("unknown", "unknown", "unknown.json")


def _family(
    family_key: str,
    output_shape: str,
    output_filename: str,
    is_metadata: bool = False,
) -> dict[str, Any]:
    return {
        "family_key": family_key,
        "output_shape": output_shape,
        "output_filename": output_filename,
        "output_file_key": f"{family_key}/{output_shape}",
        "is_metadata": is_metadata,
    }


def _volume_orderflow_shape(data_type: str, shape_source: str) -> str:
    if "cvd" in data_type and ("candlestick_derived" in shape_source or "derived" in shape_source):
        return "cvd_candlestick_derived"
    if "cvd" in data_type:
        return "cvd_time_series"
    if "bar" in shape_source:
        return "volume_bar"
    return "volume_features"


def _institutional_shape(shape_source: str) -> str:
    if "candlestick_derived" in shape_source or "derived" in shape_source:
        return "candlestick_derived"
    if "event" in shape_source:
        return "event_list"
    if "bar" in shape_source:
        return "bar"
    return "time_series"


def _open_interest_shape(shape_source: str) -> str:
    if "regime" in shape_source:
        return "regimes"
    if "candlestick_derived" in shape_source or "derived" in shape_source:
        return "candlestick_derived"
    return "time_series"


def _sentiment_shape(shape_source: str) -> str:
    if "candlestick_derived" in shape_source or "derived" in shape_source:
        return "candlestick_derived"
    if "bar" in shape_source:
        return "bar"
    return "time_series"


def _network_or_onchain_shape(shape_source: str) -> str:
    if "regime" in shape_source:
        return "regimes"
    if "event" in shape_source:
        return "event_list"
    if "bar" in shape_source:
        return "bars"
    return "time_series"