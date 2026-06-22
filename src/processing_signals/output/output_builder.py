from __future__ import annotations

from pathlib import Path
from typing import Any
import json

import pandas as pd


class OutputBuilder:
    """
    Output layer.

    Creates a JSON-safe bundle:
      - summary
      - per-source blocks
      - HMI payload references
      - ML feature snapshots
      - advanced algorithm payload hints
    """

    def __init__(self, max_rows: int | None = None):
        self.max_rows = max_rows

    def build(self, blocks: list[dict[str, Any]]) -> dict[str, Any]:
        serializable_blocks = [self._to_json_safe(block) for block in blocks]
        return {
            "pipeline": "Processing-Signals MainPipeline",
            "version": "0.1.0",
            "summary": {
                "records_processed": len(serializable_blocks),
                "data_types": self._count_data_types(serializable_blocks),
            },
            "blocks": serializable_blocks,
            "ml_feature_matrix_preview": self._build_ml_feature_matrix_preview(serializable_blocks),
        }

    def write_json(self, payload: dict[str, Any], output_path: Path) -> None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2, default=str)

    def _to_json_safe(self, value: Any) -> Any:
        if isinstance(value, pd.DataFrame):
            df = value.copy()
            if self.max_rows is not None:
                df = df.tail(self.max_rows)
            return df.to_dict(orient="records")

        if isinstance(value, pd.Series):
            s = value.copy()
            if self.max_rows is not None:
                s = s.tail(self.max_rows)
            return s.to_list()

        if isinstance(value, dict):
            return {str(k): self._to_json_safe(v) for k, v in value.items()}

        if isinstance(value, list):
            return [self._to_json_safe(v) for v in value]

        return value

    @staticmethod
    def _count_data_types(blocks: list[dict[str, Any]]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for block in blocks:
            data_type = block.get("detected", {}).get("data_type", "unknown")
            counts[data_type] = counts.get(data_type, 0) + 1
        return counts

    @staticmethod
    def _build_ml_feature_matrix_preview(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for block in blocks:
            routes = block.get("routes", {})
            if not routes.get("ml", {}).get("include_feature_vector"):
                continue

            feature_snapshot = block.get("math", {}).get("feature_snapshot", {})
            row = {
                "source_name": block.get("source_name"),
                "data_type": block.get("detected", {}).get("data_type"),
                "symbol": block.get("detected", {}).get("symbol"),
                "timeframe": block.get("detected", {}).get("timeframe"),
            }
            row.update(feature_snapshot)
            rows.append(row)
        return rows
