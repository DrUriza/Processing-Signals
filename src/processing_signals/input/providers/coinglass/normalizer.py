from __future__ import annotations

from typing import Any

from processing_signals.input.providers.common import coerce_number, normalize_timestamp
from processing_signals.input.providers.coinglass.registry import COINGLASS_ENDPOINTS
from processing_signals.input.schemas.normalized_record import NormalizedRecord
from processing_signals.input.schemas.raw_envelope import RawEnvelope


class CoinGlassNormalizer:
    provider = "coinglass"

    def normalize(self, envelope: RawEnvelope, symbol: str | None = None, timeframe: str | None = None) -> NormalizedRecord:
        endpoint = COINGLASS_ENDPOINTS[envelope.endpoint_id]
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
        if isinstance(payload, dict):
            data = payload.get("data", payload)
            if isinstance(data, dict):
                for key in ["list", "items", "records", "result"]:
                    if isinstance(data.get(key), list):
                        return [row for row in data[key] if isinstance(row, dict)]
            if isinstance(data, list):
                return [row for row in data if isinstance(row, dict)]
        if isinstance(payload, list):
            return [row for row in payload if isinstance(row, dict)]
        return []

    @staticmethod
    def _normalize_row(row: dict[str, Any]) -> dict[str, Any]:
        normalized = {str(key): coerce_number(value) for key, value in row.items()}
        timestamp = normalized.get("timestamp", normalized.get("time", normalized.get("t")))
        normalized["timestamp"] = normalize_timestamp(timestamp)
        return normalized
