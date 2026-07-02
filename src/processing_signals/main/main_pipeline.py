from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from processing_signals.classification.output_classifier import OutputClassifier
from processing_signals.input.config import RuntimeConfig, load_runtime_config
from processing_signals.input.input_reader import InputReader
from processing_signals.input.json_loader import InputRecord
from processing_signals.input.input_pipeline import InputPipeline
from processing_signals.output.family_output_builder import FamilyOutputBuilder
from processing_signals.output.output_builder import OutputBuilder
from processing_signals.output.output_family_rules import resolve_output_family
from processing_signals.output.output_validator import OutputValidator
from processing_signals.processing.detection.data_type_detector import DataTypeDetector
from processing_signals.processing.indicator_decision_engine import IndicatorDecisionEngine
from processing_signals.processing.transforms.transform_engine import TransformEngine
from processing_signals.processing.vectorization.vectorizer import Vectorizer
from processing_signals.processing.math.math_engine import ProcessingMathEngine
from processing_signals.processing.normalization.normalizer import Normalizer
from processing_signals.processing.patterns.pattern_engine import PatternEngine


class MainPipeline:
    """Orchestrates input, processing, math, patterns, classification, and output."""

    def __init__(
        self,
        input_path: Path,
        output_path: Path,
        max_rows: int | None = None,
        write_validation_report: bool = False,
        write_manifest: bool = False,
    ):
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.write_validation_report = write_validation_report
        self.write_manifest = write_manifest
        self.loader = InputReader(self.input_path)
        self.detector = DataTypeDetector()
        self.normalizer = Normalizer()
        self.vectorizer = Vectorizer()
        self.transform_engine = TransformEngine()
        self.decision_engine = IndicatorDecisionEngine()
        self.math_engine = ProcessingMathEngine()
        self.pattern_engine = PatternEngine()
        self.classifier = OutputClassifier()
        self.output_builder = OutputBuilder(max_rows=max_rows)

    def run(self) -> dict[str, Any]:
        raw_payloads = self.loader.load()
        detected_blocks = self._detect_blocks(raw_payloads)
        normalized_blocks = self._normalize_blocks(detected_blocks)
        vectorized_blocks = self._vectorize_blocks(normalized_blocks)
        transformed_blocks = self._transform_blocks(vectorized_blocks)
        math_blocks = self._run_math(transformed_blocks)
        pattern_blocks = self._run_patterns(math_blocks)
        blocks = self._classify_blocks(pattern_blocks)
        family_output_dir = self.output_path.parent / "families"

        for block in blocks:
            family_info = resolve_output_family(block)
            block.update(family_info)
            if family_info.get("is_metadata"):
                block["family_output_path"] = (
                    str(self.output_path.parent / "metadata" / family_info["output_filename"])
                    if self.write_manifest
                    else ""
                )
            else:
                block["family_output_path"] = str(
                    family_output_dir / family_info["family_key"] / family_info["output_filename"]
                )

        family_builder = FamilyOutputBuilder(
            output_dir=family_output_dir,
            pipeline_name="Processing-Signals MainPipeline",
            version="0.1.0",
        )
        family_outputs_index = family_builder.write_family_outputs(blocks)

        payload = self.output_builder.build(blocks)
        payload["family_outputs"] = family_outputs_index
        payload["official_families"] = family_outputs_index.get("official_families", [])
        payload["active_families"] = family_outputs_index.get("active_families", [])
        payload["inactive_families"] = family_outputs_index.get("inactive_families", [])
        manifest = self.output_builder.build_manifest(blocks)
        validation_report = OutputValidator(self.output_path.parent).validate()
        payload["manifest_summary"] = {
            "output_shape": manifest["output_shape"],
            "records_processed": manifest["records_processed"],
        }
        payload["validation"] = validation_report
        payload["validation_status"] = validation_report["status"]
        payload["errors"] = validation_report.get("errors", [])
        payload["warnings"] = [
            *payload.get("warnings", []),
            *validation_report.get("warnings", []),
        ]

        if self.write_manifest:
            manifest_path = self.output_path.parent / "metadata" / "manifest.json"
            self.output_builder.write_json(manifest, manifest_path)
            payload["manifest_summary"]["path"] = str(manifest_path)
            payload["metaoutputs"] = [
                {
                    "output_shape": "manifest",
                    "path": str(manifest_path),
                    "records_processed": manifest["records_processed"],
                }
            ]

        if self.write_validation_report:
            validation_path = self.output_path.parent / "validation_report.json"
            self.output_builder.write_json(validation_report, validation_path)
            payload["validation_report"] = str(validation_path)

        self.output_builder.write_json(payload, self.output_path)
        return payload

    def _detect_blocks(self, raw_payloads: list[InputRecord]) -> list[dict[str, Any]]:
        return [
            {
                "source_name": record.source_name,
                "raw_payload": record.payload,
                "detected": self.detector.detect(record.payload, source_name=record.source_name),
            }
            for record in raw_payloads
        ]

    def _normalize_blocks(self, detected_blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        for block in detected_blocks:
            block["normalized"] = self.normalizer.normalize(block["raw_payload"], block["detected"])
        return detected_blocks

    def _vectorize_blocks(self, normalized_blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        for block in normalized_blocks:
            block["vectorized"] = self.vectorizer.vectorize(block["normalized"], block["detected"])
        return normalized_blocks

    def _transform_blocks(self, vectorized_blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        for block in vectorized_blocks:
            block["transforms"] = self.transform_engine.transform(
                block["normalized"],
                block["detected"],
                block["vectorized"],
            )
        return vectorized_blocks

    def _run_math(self, transformed_blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        for block in transformed_blocks:
            decision = self.decision_engine.decide(block["detected"], block["normalized"])
            block["decision"] = decision
            block["math"] = self.math_engine.compute(block["normalized"], decision)
            block["view_math"] = self.math_engine.compute_view_math(block["transforms"])
        return transformed_blocks

    def _run_patterns(self, math_blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        for block in math_blocks:
            block["patterns"] = self.pattern_engine.detect(
                block["normalized"],
                block["math"],
                block["decision"],
                transforms=block.get("transforms", {}),
                view_math=block.get("view_math", {}),
            )
        return math_blocks

    def _classify_blocks(self, pattern_blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        for block in pattern_blocks:
            block["routes"] = self.classifier.classify(
                block["detected"],
                block["normalized"],
                block["math"],
                block["patterns"],
                block["decision"],
            )
            block.pop("raw_payload", None)
        return pattern_blocks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Processing-Signals main pipeline: input -> processing -> math -> classification -> output."
    )
    parser.add_argument(
        "--runtime",
        default="runtime.json",
        help="Runtime configuration JSON used to orchestrate the internal input pipeline before processing.",
    )
    parser.add_argument(
        "--input",
        default=None,
        help="Input JSON file, ZIP file, or directory containing JSON files. Defaults to the first supported file in data_input/.",
    )
    parser.add_argument(
        "--output",
        default="src/processing_signals/output/main_pipeline_output.json",
        help="Output JSON path.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional row limit for small previews in the master report.",
    )
    parser.add_argument(
        "--write-validation-report",
        action="store_true",
        help="Write src/processing_signals/output/validation_report.json in addition to embedding validation in the main output.",
    )
    parser.add_argument(
        "--write-manifest",
        action="store_true",
        help="Write src/processing_signals/output/metadata/manifest.json in addition to embedding manifest in the main output.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runtime = _load_runtime_if_exists(args.runtime)
    input_path = Path(args.input) if args.input else None

    if runtime and runtime.input.enabled and not args.input:
        input_result = InputPipeline(runtime_path=args.runtime).run()
        input_path = Path(input_result["output_path"])
    elif input_path is None:
        input_path = resolve_input_path(None)

    output_path = Path(args.output)
    if runtime and args.output == "src/processing_signals/output/main_pipeline_output.json":
        output_path = Path(runtime.processing.output_path)

    pipeline = MainPipeline(
        input_path=input_path,
        output_path=output_path,
        max_rows=args.max_rows,
        write_validation_report=args.write_validation_report,
        write_manifest=args.write_manifest,
    )
    result = pipeline.run()
    print("input:")
    print(input_path.resolve())
    print()
    print(f"records_processed: {result['summary']['records_processed']}")
    print()
    print("data_types:")
    for data_type, count in result["summary"]["data_types"].items():
        print(f"{data_type}: {count}")
    print()
    print("main_output:")
    print(output_path.resolve())
    print()
    print(f"validation_status: {result.get('validation_status')}")
    print()
    if args.write_validation_report:
        print("validation_report:")
        print(output_path.parent / "validation_report.json")
        print()
    if args.write_manifest:
        print("metaoutputs:")
        for output in result.get("metaoutputs", []):
            print(output["path"])
        print()
    print("official_families:")
    for family_key in result.get("family_outputs", {}).get("official_families", []):
        print(family_key)
    print()
    print("active_families:")
    for family_key in result.get("family_outputs", {}).get("active_families", []):
        print(family_key)
    print()
    print("inactive_families:")
    for family_key in result.get("family_outputs", {}).get("inactive_families", []):
        print(family_key)
    print()
    print("family_outputs:")
    for family in result.get("family_outputs", {}).get("families", []):
        print(f"{family['family_key']}:")
        for output in family.get("outputs", []):
            print(output["path"])
        print()


def resolve_input_path(input_arg: str | None) -> Path:
    if input_arg:
        return Path(input_arg)

    input_dir = Path("data_input")
    if not input_dir.exists():
        raise FileNotFoundError("No --input provided and data_input/ does not exist.")

    candidates = [
        path
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".zip", ".json"}
    ]
    candidates.sort(key=lambda path: (path.suffix.lower() != ".zip", path.name.lower()))

    if not candidates:
        raise FileNotFoundError("No --input provided and no .zip or .json files were found in data_input/.")

    return candidates[0]


def _load_runtime_if_exists(path_str: str | None) -> RuntimeConfig | None:
    if not path_str:
        return None

    path = Path(path_str)
    if not path.exists():
        return None

    return load_runtime_config(path)


if __name__ == "__main__":
    main()

