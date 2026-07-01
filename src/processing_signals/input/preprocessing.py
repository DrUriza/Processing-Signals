"""Preprocess and normalize extracted Input payloads."""

from __future__ import annotations

from typing import Any


def ensure_unix_seconds(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for record in records:
        item = dict(record)
        timestamp = item.get("timestamp")
        if isinstance(timestamp, (int, float)) and timestamp > 10_000_000_000:
            item["timestamp"] = int(timestamp // 1000)
        output.append(item)
    return output


def cast_numeric_strings(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for record in records:
        item: dict[str, Any] = {}
        for key, value in record.items():
            if isinstance(value, str):
                try:
                    item[key] = float(value)
                    continue
                except ValueError:
                    pass
            item[key] = value
        output.append(item)
    return output


def drop_duplicate_timestamps(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[Any] = set()
    output: list[dict[str, Any]] = []
    for record in records:
        timestamp = record.get("timestamp")
        if timestamp in seen:
            continue
        seen.add(timestamp)
        output.append(record)
    return output


def sort_by_timestamp(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(records, key=lambda item: item.get("timestamp", 0))


def evaluate_quality(records: list[dict[str, Any]], min_required: int, duplicates_removed: int) -> dict[str, Any]:
    status = "ok" if len(records) >= min_required else "warning"
    return {
        "status": status,
        "records_count": len(records),
        "min_records_required": min_required,
        "duplicates_removed": duplicates_removed,
    }


def normalize_extracted_raw(provider: str, run_id: str, raw_payloads: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalize raw payloads for one provider."""
    normalized: list[dict[str, Any]] = []
    for raw in raw_payloads:
        if raw.get("status") != "ok":
            continue
        records = _sorted_unique(raw.get("records", []))
        normalized.append(
            {
                "schema_version": "1.0",
                "run_id": run_id,
                "provider": provider,
                "family": raw["family"],
                "subtype": raw["subtype"],
                "data_type": raw["data_type"],
                "asset": raw["asset"],
                "symbol": raw["symbol"],
                "timeframe": raw.get("timeframe"),
                "extraction_window": raw.get("extraction_window"),
                "records": records,
                "metadata": {
                    "source": "synthetic" if raw.get("metadata", {}).get("mode") == "synthetic" else "api",
                    "provider_endpoint": raw["endpoint_name"],
                    "path": raw.get("metadata", {}).get("path"),
                },
                "quality": {
                    "status": "ok" if records else "warning",
                    "records_count": len(records),
                    "min_records_required": 0 if raw.get("timeframe") is None else len(records),
                },
            }
        )
    return normalized


def _sorted_unique(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[Any] = set()
    output: list[dict[str, Any]] = []
    for record in sorted(records, key=lambda item: item.get("timestamp", 0)):
        timestamp = record.get("timestamp")
        if timestamp in seen:
            continue
        seen.add(timestamp)
        output.append(record)
    return output
