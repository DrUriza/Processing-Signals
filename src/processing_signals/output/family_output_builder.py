from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import shutil
from typing import Any

import pandas as pd


class FamilyOutputBuilder:
    OFFICIAL_FAMILIES = [
        "prices_ohlcv",
        "volume_orderflow",
        "liquidity_microstructure",
        "institutional_flows",
        "liquidations",
        "derivatives_open_interest",
        "sentiment_positioning",
        "mining_network_health",
        "onchain_holder_behavior",
    ]

    def __init__(self, output_dir: Path, pipeline_name: str, version: str):
        self.output_dir = Path(output_dir)
        self.metadata_dir = self.output_dir.parent / "metadata"
        self.pipeline_name = pipeline_name
        self.version = version

    def write_family_outputs(self, blocks: list[dict[str, Any]]) -> dict[str, Any]:
        self._prepare_output_roots()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)

        for block in blocks:
            family_key = block.get("family_key", "unknown")
            output_shape = block.get("output_shape", "unknown")

            if block.get("is_metadata"):
                block["family_output_path"] = ""
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
                for variant_shape, variant_filename in self._volume_derivative_variants(block):
                    output_path = self.output_dir / "volume_orderflow" / variant_filename
                    block["family_output_paths"].append(str(output_path))
                    grouped[("volume_orderflow", variant_shape)].append(block)
                block["family_output_path"] = block["family_output_paths"][0] if block["family_output_paths"] else ""
            elif family_key == "volume_orderflow":
                block["family_output_paths"] = []
                for variant_shape, variant_filename in self._volume_orderflow_variants(block):
                    output_path = self.output_dir / family_key / variant_filename
                    block["family_output_paths"].append(str(output_path))
                    grouped[(family_key, variant_shape)].append(block)
                block["family_output_path"] = block["family_output_paths"][0] if block["family_output_paths"] else ""
            elif family_key in {"institutional_flows", "liquidations"}:
                block["family_output_paths"] = []
                for variant_shape, variant_filename in self._four_shape_variants():
                    output_path = self.output_dir / family_key / variant_filename
                    block["family_output_paths"].append(str(output_path))
                    grouped[(family_key, variant_shape)].append(block)
                block["family_output_path"] = block["family_output_paths"][0] if block["family_output_paths"] else ""
            elif family_key == "derivatives_open_interest":
                block["family_output_paths"] = []
                for variant_shape, variant_filename in self._open_interest_variants():
                    output_path = self.output_dir / family_key / variant_filename
                    block["family_output_paths"].append(str(output_path))
                    grouped[(family_key, variant_shape)].append(block)
                block["family_output_path"] = block["family_output_paths"][0] if block["family_output_paths"] else ""
            elif family_key == "sentiment_positioning":
                block["family_output_paths"] = []
                for variant_shape, variant_filename in self._sentiment_variants():
                    output_path = self.output_dir / family_key / variant_filename
                    block["family_output_paths"].append(str(output_path))
                    grouped[(family_key, variant_shape)].append(block)
                block["family_output_path"] = block["family_output_paths"][0] if block["family_output_paths"] else ""
            elif family_key in {"mining_network_health", "onchain_holder_behavior"}:
                block["family_output_paths"] = []
                for variant_shape, variant_filename in self._network_onchain_variants():
                    output_path = self.output_dir / family_key / variant_filename
                    block["family_output_paths"].append(str(output_path))
                    grouped[(family_key, variant_shape)].append(block)
                block["family_output_path"] = block["family_output_paths"][0] if block["family_output_paths"] else ""
            else:
                output_filename = block.get("output_filename", f"{output_shape}.json")
                output_path = self.output_dir / family_key / output_filename
                block["family_output_path"] = str(output_path)
                grouped[(family_key, output_shape)].append(block)

        self._validate_family_output_limits(grouped)

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

        active_families = [family for family in self.OFFICIAL_FAMILIES if family in family_entries]
        inactive_families = [family for family in self.OFFICIAL_FAMILIES if family not in family_entries]

        return {
            "families_root": str(self.output_dir),
            "official_families": self.OFFICIAL_FAMILIES,
            "active_families": active_families,
            "inactive_families": inactive_families,
            "families": [
                {"family_key": family_key, "outputs": family_entries[family_key]}
                for family_key in self.OFFICIAL_FAMILIES
                if family_key in family_entries
            ],
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
        block["source_subtype"] = subtype

        shapes = ["orderbook", "time_series", "bars"]
        if subtype in {"large_trades", "whale_orders"}:
            shapes.append("event_list")
        return [(shape, f"{shape}.json") for shape in shapes]

    @staticmethod
    def _prices_variants(block: dict[str, Any]) -> list[tuple[str, str]]:
        if block.get("detected", {}).get("data_type") != "candlestick":
            return []
        return [
            ("candlestick", "candlestick.json"),
            ("time_series", "time_series.json"),
        ]

    @staticmethod
    def _volume_derivative_variants(block: dict[str, Any]) -> list[tuple[str, str]]:
        if block.get("detected", {}).get("data_type") != "candlestick":
            return []
        normalized = block.get("normalized", {})
        dataframe = normalized.get("dataframe")
        if not isinstance(dataframe, pd.DataFrame):
            return []
        if "volume" not in dataframe.columns and "notional_volume" not in dataframe.columns:
            return []
        return [("volume_features", "volume_features.json")]

    @staticmethod
    def _volume_orderflow_variants(block: dict[str, Any]) -> list[tuple[str, str]]:
        if block.get("detected", {}).get("data_type") != "cvd":
            return []
        return [
            ("cvd_time_series", "cvd_time_series.json"),
            ("cvd_candlestick_derived", "cvd_candlestick_derived.json"),
            ("orderflow_features", "orderflow_features.json"),
        ]

    @staticmethod
    def _four_shape_variants() -> list[tuple[str, str]]:
        return [
            ("time_series", "time_series.json"),
            ("bars", "bars.json"),
            ("event_list", "event_list.json"),
            ("candlestick_derived", "candlestick_derived.json"),
        ]

    @staticmethod
    def _open_interest_variants() -> list[tuple[str, str]]:
        return [
            ("time_series", "time_series.json"),
            ("candlestick_derived", "candlestick_derived.json"),
            ("regimes", "regimes.json"),
        ]

    @staticmethod
    def _sentiment_variants() -> list[tuple[str, str]]:
        return [
            ("time_series", "time_series.json"),
            ("bars", "bars.json"),
            ("candlestick_derived", "candlestick_derived.json"),
        ]

    @staticmethod
    def _network_onchain_variants() -> list[tuple[str, str]]:
        return [
            ("time_series", "time_series.json"),
            ("bars", "bars.json"),
            ("event_list", "event_list.json"),
            ("regimes", "regimes.json"),
        ]

    @classmethod
    def _validate_family_output_limits(cls, grouped: dict[tuple[str, str], list[dict[str, Any]]]) -> None:
        counts: dict[str, set[str]] = defaultdict(set)
        for family_key, output_shape in grouped:
            counts[family_key].add(output_shape)

        too_many = {
            family_key: sorted(output_shapes)
            for family_key, output_shapes in counts.items()
            if family_key in cls.OFFICIAL_FAMILIES and len(output_shapes) > 4
        }
        if too_many:
            details = "; ".join(f"{family}: {shapes}" for family, shapes in sorted(too_many.items()))
            raise ValueError(f"Family output contract violation: more than 4 outputs generated for {details}")

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
