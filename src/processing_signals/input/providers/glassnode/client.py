from __future__ import annotations

import os
from typing import Any

from processing_signals.input.providers.common import load_dotenv, request_json
from processing_signals.input.providers.glassnode.registry import GLASSNODE_ENDPOINTS
from processing_signals.input.schemas.provider_status import ProviderStatus
from processing_signals.input.schemas.raw_envelope import RawEnvelope


class GlassnodeClient:
    provider = "glassnode"
    base_url = "https://api.glassnode.com"
    api_key_env = "GLASSNODE_API_KEY"

    def __init__(self, api_key: str | None = None, base_url: str | None = None):
        load_dotenv()
        self.api_key = api_key or os.getenv(self.api_key_env)
        self.base_url = base_url or self.base_url

    def status(self) -> ProviderStatus:
        return ProviderStatus(
            provider=self.provider,
            configured=bool(self.api_key),
            missing_env=[] if self.api_key else [self.api_key_env],
            endpoints=sorted(GLASSNODE_ENDPOINTS),
        )

    def fetch(self, endpoint_id: str, params: dict[str, Any] | None = None) -> RawEnvelope:
        endpoint = GLASSNODE_ENDPOINTS[endpoint_id]
        request_params = {**(params or {}), "api_key": self.api_key}
        return request_json(self.provider, endpoint_id, self.base_url, endpoint["path"], params=request_params)
