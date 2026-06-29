from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from processing_signals.input.schemas.raw_envelope import RawEnvelope


def load_dotenv(path: Path | str = ".env") -> None:
    env_path = Path(path)
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def coerce_number(value: Any) -> Any:
    if isinstance(value, str):
        text = value.strip().replace(",", "")
        if text == "":
            return value
        try:
            return float(text)
        except ValueError:
            return value
    return value


def normalize_timestamp(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        if value > 10_000_000_000:
            return int(value / 1000)
        return int(value)
    return value


def request_json(
    provider: str,
    endpoint_id: str,
    base_url: str,
    path: str,
    headers: dict[str, str] | None = None,
    params: dict[str, Any] | None = None,
    timeout: int = 30,
) -> RawEnvelope:
    query = urlencode({key: value for key, value in (params or {}).items() if value is not None})
    url = f"{base_url.rstrip('/')}/{path.lstrip('/')}"
    if query:
        url = f"{url}?{query}"

    request = Request(url, headers=headers or {})
    try:
        with urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
            status_code = int(response.status)
            return RawEnvelope(provider, endpoint_id, url, status_code, 200 <= status_code < 300, payload=payload)
    except HTTPError as exc:
        text = exc.read().decode("utf-8", errors="replace")
        return RawEnvelope(provider, endpoint_id, url, int(exc.code), False, error=text)
    except (URLError, TimeoutError, json.JSONDecodeError) as exc:
        return RawEnvelope(provider, endpoint_id, url, 0, False, error=f"{type(exc).__name__}: {exc}")
