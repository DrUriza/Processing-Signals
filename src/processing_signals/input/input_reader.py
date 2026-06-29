from __future__ import annotations

from pathlib import Path

from processing_signals.input.json_loader import InputRecord, JsonInputLoader
from processing_signals.input.zip_loader import ZipLoader


class InputReader:
    """Route raw input paths to the correct low-level loader."""

    def __init__(self, input_path: Path):
        self.input_path = Path(input_path)

    def load(self) -> list[InputRecord]:
        if not self.input_path.exists():
            raise FileNotFoundError(f"Input path not found: {self.input_path}")

        if self.input_path.is_dir() or self.input_path.suffix.lower() == ".json":
            return JsonInputLoader(self.input_path).load()

        if self.input_path.suffix.lower() == ".zip":
            return ZipLoader(self.input_path).load()

        raise ValueError(f"Unsupported input type: {self.input_path}")


__all__ = ["InputReader", "InputRecord"]
