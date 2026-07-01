"""Connection creation for live and synthetic Input modes."""

from __future__ import annotations

from typing import Any


def create_connection(provider: str, mode: str, endpoint_config: dict[str, Any]) -> dict[str, Any]:
    """Create a lightweight connection descriptor for an endpoint."""
    if mode not in {"synthetic", "live"}:
        raise ValueError(f"Unsupported input mode: {mode}")

    return {
        "provider": provider,
        "mode": mode,
        "base_url": endpoint_config.get("base_url"),
        "path": endpoint_config.get("path"),
        "callable_live": mode == "live"
        and endpoint_config.get("path") is not None
        and endpoint_config.get("live_status") != "external_provider_required",
    }
