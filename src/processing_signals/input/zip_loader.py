from __future__ import annotations

import json
from pathlib import Path
from zipfile import ZipFile

from processing_signals.input.json_loader import InputRecord


class ZipLoader:
    """Load raw JSON payloads from a ZIP archive."""

    def __init__(self, zip_path: Path):
        self.zip_path = Path(zip_path)

    def load(self) -> list[InputRecord]:
        if not self.zip_path.exists():
            raise FileNotFoundError(f"ZIP path not found: {self.zip_path}")
        if self.zip_path.suffix.lower() != ".zip":
            raise ValueError(f"Expected a .zip input path: {self.zip_path}")

        records: list[InputRecord] = []
        with ZipFile(self.zip_path, "r") as archive:
            for name in sorted(archive.namelist()):
                if not name.lower().endswith(".json"):
                    continue
                with archive.open(name) as fh:
                    payload = json.load(fh)
                if not isinstance(payload, dict):
                    raise ValueError(f"Expected a JSON object in ZIP member: {name}")
                records.append(InputRecord(source_name=Path(name).name, payload=payload))

        if not records:
            raise ValueError(f"No JSON files found in ZIP: {self.zip_path}")
        return records


__all__ = ["ZipLoader", "InputRecord"]
