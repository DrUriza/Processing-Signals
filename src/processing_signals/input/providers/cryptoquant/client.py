from __future__ import annotations

import os
from typing import Any

from processing_signals.input.providers.common import load_dotenv, request_json
from processing_signals.input.providers.cryptoquant.registry import CRYPTOQUANT_ENDPOINTS
from processing_signals.input.schemas.provider_status import ProviderStatus
from processing_signals.input.schemas.raw_envelope import RawEnvelope


class CryptoQuantClient:
    provider = "cryptoquant"
    base_url = "https://api.cryptoquant.com"
    token_env = "CRYPTOQUANT_ACCESS_TOKEN"

    def __init__(self, access_token: str | None = None, base_url: str | None = None):
        load_dotenv()
        self.access_token = access_token or os.getenv(self.token_env)
        self.base_url = base_url or self.base_url

    def status(self) -> ProviderStatus:
        return ProviderStatus(
            provider=self.provider,
            configured=bool(self.access_token),
            missing_env=[] if self.access_token else [self.token_env],
            endpoints=sorted(CRYPTOQUANT_ENDPOINTS),
        )

    def fetch(self, endpoint_id: str, params: dict[str, Any] | None = None) -> RawEnvelope:
        endpoint = CRYPTOQUANT_ENDPOINTS[endpoint_id]
        headers = {"Authorization": f"Bearer {self.access_token or ''}"}
        return request_json(self.provider, endpoint_id, self.base_url, endpoint["path"], headers=headers, params=params)
