from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import zipfile
from typing import Any


@dataclass(frozen=True)
class InputRecord:
    source_name: str
    payload: dict[str, Any]


class JsonInputLoader:
    """
    Input layer.

    Accepts:
      - a single JSON file
      - a ZIP containing JSON files
      - a directory containing JSON files

    It does not interpret trading semantics. It only loads raw JSON payloads.
    """

    def __init__(self, input_path: Path):
        self.input_path = Path(input_path)

    def load(self) -> list[InputRecord]:
        if not self.input_path.exists():
            raise FileNotFoundError(f"Input path not found: {self.input_path}")

        if self.input_path.is_dir():
            return self._load_directory(self.input_path)

        if self.input_path.suffix.lower() == ".zip":
            return self._load_zip(self.input_path)

        if self.input_path.suffix.lower() == ".json":
            payload = self._load_json_file(self.input_path)
            return [InputRecord(source_name=self.input_path.name, payload=payload)]

        raise ValueError(f"Unsupported input type: {self.input_path}")

    def _load_directory(self, directory: Path) -> list[InputRecord]:
        records: list[InputRecord] = []
        for path in sorted(directory.glob("*.json")):
            records.append(InputRecord(source_name=path.name, payload=self._load_json_file(path)))
        if not records:
            raise ValueError(f"No JSON files found in directory: {directory}")
        return records

    def _load_zip(self, zip_path: Path) -> list[InputRecord]:
        records: list[InputRecord] = []
        with zipfile.ZipFile(zip_path, "r") as archive:
            for name in sorted(archive.namelist()):
                if not name.lower().endswith(".json"):
                    continue
                with archive.open(name) as fh:
                    payload = json.load(fh)
                records.append(InputRecord(source_name=Path(name).name, payload=payload))

        if not records:
            raise ValueError(f"No JSON files found in ZIP: {zip_path}")
        return records

    @staticmethod
    def _load_json_file(path: Path) -> dict[str, Any]:
        with path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
        if not isinstance(payload, dict):
            raise ValueError(f"Expected a JSON object in {path}")
        return payload
