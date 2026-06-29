from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ProviderStatus:
    provider: str
    configured: bool
    missing_env: list[str] = field(default_factory=list)
    endpoints: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
