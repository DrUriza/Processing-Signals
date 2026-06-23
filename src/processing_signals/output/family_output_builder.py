from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import shutil
from typing import Any

import pandas as pd


class FamilyOutputBuilder:
    OFFICIAL_FAMILIES = {
        "prices_ohlcv",
        "volume_orderflow",
        "liquidity_microstructure",
        "institutional_flows",
        "liquidations",
        "derivatives_open_interest",
        "sentiment_positioning",
    }

    def __init__(self, output_dir: Path, pipeline_name: str, version: str):
        self.output_dir = Path(output_dir)
        self.metadata_dir = self.output_dir.parent / "metadata"
        self.pipeline_name = pipeline_name
        self.version = version

    def write_family_outputs(self, blocks: list[dict[str, Any]]) -> dict[str, Any]:
        self._prepare_output_roots()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        metadata_outputs: list[dict[str, Any]] = []

        for block in blocks:
            family_key = block.get("family_key", "unknown")
            output_shape = block.get("output_shape", "unknown")

            if block.get("is_metadata"):
                output_filename = block.get("output_filename", f"{output_shape}.json")
                output_path = self.metadata_dir / output_filename
                block["family_output_path"] = str(output_path)
                payload = self._build_metadata_payload(output_shape, [block])
                self._write_json(payload, output_path)
                metadata_outputs.append(
                    {
                        "output_shape": output_shape,
                        "path": str(output_path),
                        "records_processed": 1,
                    }
                )
                continue

            if family_key not in self.OFFICIAL_FAMILIES:
                continue

            if family_key == "liquidity_microstructure":
                block["family_output_paths"] = []
                for variant_shape, variant_filename in self._liquidity_variants(block):
                    output_path = self.output_dir / family_key / variant_filename
                    block["family_output_paths"].append(str(output_path))
                    grouped[(family_key, variant_shape)].append(block)
                block["family_output_path"] = block["family_output_paths"][0] if block["family_output_paths"] else ""
            elif family_key == "prices_ohlcv":
                block["family_output_paths"] = []
                for variant_shape, variant_filename in self._prices_variants(block):
                    output_path = self.output_dir / family_key / variant_filename
                    block["family_output_paths"].append(str(output_path))
                    grouped[(family_key, variant_shape)].append(block)
                block["family_output_path"] = block["family_output_paths"][0] if block["family_output_paths"] else ""
            else:
                output_filename = block.get("output_filename", f"{output_shape}.json")
                output_path = self.output_dir / family_key / output_filename
                block["family_output_path"] = str(output_path)
                grouped[(family_key, output_shape)].append(block)

        family_entries: dict[str, list[dict[str, Any]]] = defaultdict(list)

        for (family_key, output_shape), family_blocks in sorted(grouped.items()):
            family_dir = self.output_dir / family_key
            family_dir.mkdir(parents=True, exist_ok=True)
            output_filename = f"{output_shape}.json"
            output_path = family_dir / output_filename
            payload = self._build_family_payload(family_key, output_shape, family_blocks)
            self._write_json(payload, output_path)
            family_entries[family_key].append(
                {
                    "output_shape": output_shape,
                    "path": str(output_path),
                    "records_processed": len(family_blocks),
                }
            )

        return {
            "families_root": str(self.output_dir),
            "metadata_root": str(self.metadata_dir),
            "families": [
                {"family_key": family_key, "outputs": outputs}
                for family_key, outputs in sorted(family_entries.items())
            ],
            "metadata": metadata_outputs,
        }

    def _prepare_output_roots(self) -> None:
        for path in [self.output_dir, self.metadata_dir]:
            if path.exists():
                shutil.rmtree(path)

    def _build_family_payload(
        self,
        family_key: str,
        output_shape: str,
        blocks: list[dict[str, Any]],
    ) -> dict[str, Any]:
        serializable_blocks = [self._to_json_safe(block) for block in blocks]
        return {
            "pipeline": self.pipeline_name,
            "version": self.version,
            "family_key": family_key,
            "output_shape": output_shape,
            "records_processed": len(serializable_blocks),
            "symbols": self._unique_values(serializable_blocks, "symbol"),
            "timeframes": self._unique_values(serializable_blocks, "timeframe"),
            "data_types": self._count_data_types(serializable_blocks),
            "blocks": serializable_blocks,
            "ml_feature_matrix_preview": self._build_ml_feature_matrix_preview(serializable_blocks),
        }

    def _build_metadata_payload(self, output_shape: str, blocks: list[dict[str, Any]]) -> dict[str, Any]:
        serializable_blocks = [self._to_json_safe(block) for block in blocks]
        return {
            "pipeline": self.pipeline_name,
            "version": self.version,
            "output_shape": output_shape,
            "records_processed": len(serializable_blocks),
            "blocks": serializable_blocks,
        }

    @staticmethod
    def _liquidity_variants(block: dict[str, Any]) -> list[tuple[str, str]]:
        data_type = block.get("detected", {}).get("data_type")
        subtype_by_data_type = {
            "orderbook_conventional": "conventional",
            "orderbook_large_trades": "large_trades",
            "orderbook_whale_orders": "whale_orders",
        }
        subtype = subtype_by_data_type.get(data_type)
        if subtype is None:
            return []

        shapes = ["orderbook", "time_series", "event_list", "bars"]
        return [(f"{subtype}_{shape}", f"{subtype}_{shape}.json") for shape in shapes]

    @staticmethod
    def _prices_variants(block: dict[str, Any]) -> list[tuple[str, str]]:
        if block.get("detected", {}).get("data_type") != "candlestick":
            return []
        return [
            ("candlestick", "candlestick.json"),
            ("time_series", "time_series.json"),
        ]

    @staticmethod
    def _write_json(payload: dict[str, Any], output_path: Path) -> None:
        with output_path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2, default=str)

    def _to_json_safe(self, value: Any) -> Any:
        if isinstance(value, pd.DataFrame):
            return value.to_dict(orient="records")

        if isinstance(value, pd.Series):
            return value.to_list()

        if isinstance(value, dict):
            return {str(k): self._to_json_safe(v) for k, v in value.items()}

        if isinstance(value, list):
            return [self._to_json_safe(v) for v in value]

        return value

    @staticmethod
    def _unique_values(blocks: list[dict[str, Any]], key: str) -> list[Any]:
        values = []
        for block in blocks:
            value = block.get("detected", {}).get(key)
            if value is not None and value not in values:
                values.append(value)
        return values

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
