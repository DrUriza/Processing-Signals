from __future__ import annotations

from typing import Any

from processing_signals.input.providers.common import coerce_number, normalize_timestamp
from processing_signals.input.providers.glassnode.registry import GLASSNODE_ENDPOINTS
from processing_signals.input.schemas.normalized_record import NormalizedRecord
from processing_signals.input.schemas.raw_envelope import RawEnvelope


class GlassnodeNormalizer:
    provider = "glassnode"

    def normalize(self, envelope: RawEnvelope, symbol: str | None = "BTC", timeframe: str | None = None) -> NormalizedRecord:
        endpoint = GLASSNODE_ENDPOINTS[envelope.endpoint_id]
        rows = self._extract_rows(envelope.payload)
        records = [self._normalize_row(row) for row in rows]
        return NormalizedRecord(
            provider=self.provider,
            family=endpoint["family"],
            data_type=endpoint["data_type"],
            endpoint_id=envelope.endpoint_id,
            symbol=symbol,
            timeframe=timeframe,
            records=records,
            metadata={"raw_status_code": envelope.status_code, "raw_url": envelope.url},
        )

    def _extract_rows(self, payload: Any) -> list[dict[str, Any]]:
        if isinstance(payload, list):
            return [row for row in payload if isinstance(row, dict)]
        if isinstance(payload, dict):
            data = payload.get("data", payload.get("result", payload))
            if isinstance(data, list):
                return [row for row in data if isinstance(row, dict)]
        return []

    @staticmethod
    def _normalize_row(row: dict[str, Any]) -> dict[str, Any]:
        value = row.get("v")
        if isinstance(value, dict):
            normalized = {str(key): coerce_number(item) for key, item in value.items()}
        else:
            normalized = {"value": coerce_number(value)}
        normalized["timestamp"] = normalize_timestamp(row.get("t", row.get("timestamp")))
        return normalized
