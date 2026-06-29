from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class RawEnvelope:
    provider: str
    endpoint_id: str
    url: str
    status_code: int
    ok: bool
    payload: dict[str, Any] | list[Any] | None = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
