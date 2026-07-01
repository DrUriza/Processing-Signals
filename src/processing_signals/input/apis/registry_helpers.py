"""Helpers for provider endpoint registries."""

from __future__ import annotations

from typing import Any


def endpoint(
    *,
    provider: str,
    family: str,
    subtype: str,
    path: str | None,
    coverage_role: str,
    live_status: str,
    data_type: str = "time_series",
    supports_timeframe: bool = True,
    synthetic_status: str = "supported",
    response_shape: str | None = None,
    live_supported_timeframes: list[str] | None = None,
    synthetic_timeframes: list[str] | None = None,
    extraction_windows: list[str] | None = None,
    default_params: dict[str, Any] | None = None,
    notes: str = "",
    external_provider: str | None = None,
) -> dict[str, Any]:
    if path is None and live_status in {"supported"}:
        raise ValueError(f"{provider}/{family}/{subtype} is supported but has no live path")

    synthetic_timeframes = list(synthetic_timeframes or [])
    if synthetic_status == "skip" or not supports_timeframe:
        synthetic_timeframes = []

    if extraction_windows is None:
        if data_type == "snapshot":
            extraction_windows = ["latest"]
        elif data_type in {"event_list", "heatmap"}:
            extraction_windows = ["24h"]
        else:
            extraction_windows = []

    template = None
    if synthetic_status != "skip":
        slot = "timeframe" if supports_timeframe else "extraction_window"
        template = f"{family}/{subtype}/{{{slot}}}_raw.json"

    item = {
        "provider": provider,
        "family": family,
        "subtype": subtype,
        "endpoint_name": f"{provider}_{family}_{subtype}",
        "path": path,
        "method": "GET" if path else None,
        "coverage_role": coverage_role,
        "live_status": live_status,
        "synthetic_status": synthetic_status,
        "data_type": data_type,
        "supports_timeframe": supports_timeframe,
        "live_supported_timeframes": list(live_supported_timeframes or []),
        "synthetic_timeframes": synthetic_timeframes,
        "extraction_windows": list(extraction_windows),
        "response_shape": response_shape or provider,
        "synthetic_file_template": template,
        "default_params": dict(default_params or {}),
        "notes": notes,
    }
    if external_provider:
        item["external_provider"] = external_provider
    return item
