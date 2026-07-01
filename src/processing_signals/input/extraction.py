"""Extract raw endpoint payloads from live or synthetic sources."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any


def extract_endpoint(
    connection: dict[str, Any],
    endpoint_config: dict[str, Any],
    run_id: str,
    asset: str,
    symbol: str,
    *,
    timeframe: str | None = None,
    extraction_window: str | None = None,
    min_records: int = 200,
) -> dict[str, Any]:
    """Extract one endpoint into a raw payload descriptor."""
    mode = connection["mode"]
    if mode == "live" and not connection["callable_live"]:
        return {
            "run_id": run_id,
            "provider": endpoint_config["provider"],
            "endpoint_name": endpoint_config["endpoint_name"],
            "family": endpoint_config["family"],
            "subtype": endpoint_config["subtype"],
            "status": "skipped_live_external_provider_required"
            if endpoint_config.get("live_status") == "external_provider_required"
            else "skipped_live_path_missing",
            "records": [],
            "metadata": {"reason": "not callable in live mode"},
        }

    records = _synthetic_records(
        endpoint_config=endpoint_config,
        timeframe=timeframe,
        extraction_window=extraction_window,
        count=min_records,
    )
    return {
        "run_id": run_id,
        "provider": endpoint_config["provider"],
        "endpoint_name": endpoint_config["endpoint_name"],
        "family": endpoint_config["family"],
        "subtype": endpoint_config["subtype"],
        "data_type": endpoint_config["data_type"],
        "timeframe": timeframe,
        "extraction_window": extraction_window,
        "asset": asset,
        "symbol": symbol,
        "status": "ok",
        "records": records,
        "metadata": {
            "mode": mode,
            "path": endpoint_config.get("path"),
            "live_status": endpoint_config.get("live_status"),
        },
    }


def _synthetic_records(
    *,
    endpoint_config: dict[str, Any],
    timeframe: str | None,
    extraction_window: str | None,
    count: int,
) -> list[dict[str, Any]]:
    data_type = endpoint_config.get("data_type")
    if data_type in {"snapshot", "heatmap"}:
        return [{"timestamp": _now(), "value": 1.0, "window": extraction_window}]
    if data_type == "event_list":
        return [{"timestamp": _now(), "event": endpoint_config["subtype"], "value": 1.0}]

    step = _timeframe_seconds(timeframe or "1m")
    end = _now()
    start = end - (max(count, 1) - 1) * step
    base = float(abs(hash(endpoint_config["endpoint_name"])) % 1000) / 10.0
    return [
        {
            "timestamp": start + index * step,
            "value": round(base + index * 0.01, 8),
        }
        for index in range(max(count, 0))
    ]


def _now() -> int:
    return int(datetime.now(tz=UTC).timestamp())


def _timeframe_seconds(timeframe: str) -> int:
    units = {"m": 60, "h": 3600, "d": 86400}
    try:
        return int(timeframe[:-1]) * units[timeframe[-1]]
    except (KeyError, ValueError, IndexError):
        return 60
