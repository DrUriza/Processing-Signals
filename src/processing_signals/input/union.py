"""Union normalized Input payloads and write normalized outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


OFFICIAL_FAMILIES = [
    "prices_ohlcv",
    "liquidity_microstructure",
    "volume_orderflow",
    "institutional_flows",
    "liquidations",
    "derivatives_open_interest",
    "sentiment_positioning",
    "on_chain_miners",
    "options_volatility",
]


def union_normalized_payloads(
    payloads: list[dict[str, Any]],
    run_id: str,
    *,
    output_dir: str | Path = "data_input/normalized",
) -> dict[str, Any]:
    """Write normalized payloads by family and create a manifest."""
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)

    files: list[str] = []
    family_counts: dict[str, int] = {}
    for payload in payloads:
        family = payload["family"]
        if family not in OFFICIAL_FAMILIES:
            raise ValueError(f"Unknown official family: {family}")

        family_dir = root / family
        family_dir.mkdir(parents=True, exist_ok=True)
        filename = _payload_filename(payload)
        path = family_dir / filename
        path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
        files.append(path.relative_to(root).as_posix())
        family_counts[family] = family_counts.get(family, 0) + 1

    manifest = {
        "schema_version": "1.0",
        "run_id": run_id,
        "output_dir": str(root),
        "files": files,
        "families": family_counts,
        "records_total": sum(len(payload.get("records", [])) for payload in payloads),
        "status": "ok" if payloads else "empty",
    }
    (root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=True, indent=2), encoding="utf-8")
    return manifest


def _payload_filename(payload: dict[str, Any]) -> str:
    parts = [payload["provider"], payload["subtype"]]
    if payload.get("timeframe") is not None:
        parts.append(str(payload["timeframe"]))
    elif payload.get("extraction_window") is not None:
        parts.append(str(payload["extraction_window"]))
    return "_".join(parts) + ".json"
