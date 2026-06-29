from __future__ import annotations

import os
from typing import Any

from processing_signals.input.providers.common import load_dotenv, request_json
from processing_signals.input.providers.coinglass.registry import COINGLASS_ENDPOINTS
from processing_signals.input.schemas.provider_status import ProviderStatus
from processing_signals.input.schemas.raw_envelope import RawEnvelope


class CoinGlassClient:
    provider = "coinglass"
    base_url = "https://open-api-v4.coinglass.com"
    api_key_env = "COINGLASS_API_KEY"

    def __init__(self, api_key: str | None = None, base_url: str | None = None):
        load_dotenv()
        self.api_key = api_key or os.getenv(self.api_key_env)
        self.base_url = base_url or self.base_url

    def status(self) -> ProviderStatus:
        return ProviderStatus(
            provider=self.provider,
            configured=bool(self.api_key),
            missing_env=[] if self.api_key else [self.api_key_env],
            endpoints=sorted(COINGLASS_ENDPOINTS),
        )

    def fetch(self, endpoint_id: str, params: dict[str, Any] | None = None) -> RawEnvelope:
        endpoint = COINGLASS_ENDPOINTS[endpoint_id]
        headers = {"CG-API-KEY": self.api_key or ""}
        return request_json(self.provider, endpoint_id, self.base_url, endpoint["path"], headers=headers, params=params)
