from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class NormalizedRecord:
    provider: str
    family: str
    data_type: str
    endpoint_id: str
    records: list[dict[str, Any]]
    symbol: str | None = None
    timeframe: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_processing_payload(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "family": self.family,
            "data_type": self.data_type,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "records": self.records,
            "metadata": {
                **self.metadata,
                "provider": self.provider,
                "family": self.family,
                "endpoint_id": self.endpoint_id,
            },
        }
