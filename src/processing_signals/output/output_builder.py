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
        data_types = self._count_data_types(blocks)
        payload = {
            "pipeline": "Processing-Signals MainPipeline",
            "version": "0.1.0",
            "summary": {
                "records_processed": len(blocks),
                "data_types": data_types,
            },
            "records_processed": len(blocks),
            "data_types": data_types,
            "block_index": self._build_block_index(blocks),
            "math_summary": self._build_math_summary(blocks),
            "warnings": self._collect_block_warnings(blocks),
            "errors": [],
        }
        if self.max_rows is not None:
            payload["previews"] = self._build_previews(blocks)
        return payload

    def build_manifest(self, blocks: list[dict[str, Any]]) -> dict[str, Any]:
        manifest_blocks = [block for block in blocks if block.get("is_metadata")]
        serializable_blocks = [self._to_json_safe(block) for block in manifest_blocks]
        return {
            "pipeline": "Processing-Signals MainPipeline",
            "version": "0.1.0",
            "output_shape": "manifest",
            "records_processed": len(serializable_blocks),
            "blocks": serializable_blocks,
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
    def _build_block_index(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        block_index: list[dict[str, Any]] = []
        for block in blocks:
            detected = block.get("detected", {})
            entry = {
                "source_name": block.get("source_name"),
                "data_type": detected.get("data_type"),
                "symbol": detected.get("symbol"),
                "timeframe": detected.get("timeframe"),
                "family_key": block.get("family_key"),
                "output_shape": block.get("output_shape"),
                "output_filename": block.get("output_filename"),
            }
            output_paths = block.get("family_output_paths") or []
            if output_paths:
                entry["family_output_paths"] = output_paths
            elif block.get("family_output_path"):
                entry["family_output_path"] = block.get("family_output_path")
            block_index.append(entry)
        return block_index

    @staticmethod
    def _build_math_summary(blocks: list[dict[str, Any]]) -> dict[str, Any]:
        summary = {
            "records_with_technical": 0,
            "records_with_statistics": 0,
            "records_with_statistical_regimes": 0,
            "records_with_microstructure": 0,
            "records_with_feature_snapshot": 0,
            "statistics": {
                "numeric_column_counts": {},
                "windows": [],
                "reference_columns": [],
            },
            "statistical_regimes": {
                "numeric_column_counts": {},
                "windows": [],
                "records_with_last_regimes": 0,
            },
        }

        statistics_windows: set[Any] = set()
        statistics_references: set[Any] = set()
        regime_windows: set[Any] = set()

        for block in blocks:
            math_payload = block.get("math", {})
            technical = math_payload.get("technical")
            statistics = math_payload.get("statistics") or {}
            regimes = math_payload.get("statistical_regimes") or {}
            microstructure = math_payload.get("microstructure")
            feature_snapshot = math_payload.get("feature_snapshot")

            if technical:
                summary["records_with_technical"] += 1
            if statistics:
                summary["records_with_statistics"] += 1
                for column in statistics.get("numeric_columns", []):
                    counts = summary["statistics"]["numeric_column_counts"]
                    counts[column] = counts.get(column, 0) + 1
                statistics_windows.update(statistics.get("windows", []))
                reference_column = statistics.get("reference_column")
                if reference_column:
                    statistics_references.add(reference_column)
            if regimes:
                summary["records_with_statistical_regimes"] += 1
                for column in regimes.get("numeric_columns", []):
                    counts = summary["statistical_regimes"]["numeric_column_counts"]
                    counts[column] = counts.get(column, 0) + 1
                regime_windows.update(regimes.get("windows", []))
                if regimes.get("last_regimes"):
                    summary["statistical_regimes"]["records_with_last_regimes"] += 1
            if microstructure:
                summary["records_with_microstructure"] += 1
            if feature_snapshot:
                summary["records_with_feature_snapshot"] += 1

        summary["statistics"]["windows"] = sorted(statistics_windows)
        summary["statistics"]["reference_columns"] = sorted(statistics_references)
        summary["statistical_regimes"]["windows"] = sorted(regime_windows)
        return summary

    def _build_previews(self, blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        previews: list[dict[str, Any]] = []
        for block in blocks:
            detected = block.get("detected", {})
            normalized = block.get("normalized", {})
            dataframe = normalized.get("dataframe")
            if not isinstance(dataframe, pd.DataFrame):
                continue

            preview = dataframe.tail(self.max_rows).to_dict(orient="records")
            previews.append(
                {
                    "source_name": block.get("source_name"),
                    "data_type": detected.get("data_type"),
                    "symbol": detected.get("symbol"),
                    "timeframe": detected.get("timeframe"),
                    "rows": preview,
                }
            )
        return previews

    @staticmethod
    def _collect_block_warnings(blocks: list[dict[str, Any]]) -> list[str]:
        warnings: list[str] = []
        for block in blocks:
            for warning in block.get("math", {}).get("warnings", []) or []:
                warnings.append(f"{block.get('source_name')}: {warning}")
        return warnings
