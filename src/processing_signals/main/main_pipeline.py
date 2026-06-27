from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from processing_signals.classification.output_classifier import OutputClassifier
from processing_signals.input.json_loader import JsonInputLoader
from processing_signals.output.family_output_builder import FamilyOutputBuilder
from processing_signals.output.output_builder import OutputBuilder
from processing_signals.output.output_family_rules import resolve_output_family
from processing_signals.output.output_validator import OutputValidator
from processing_signals.processing.data_type_detector import DataTypeDetector
from processing_signals.processing.indicator_decision_engine import IndicatorDecisionEngine
from processing_signals.processing.math.math_engine import ProcessingMathEngine
from processing_signals.processing.normalizer import Normalizer
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
        self.loader = JsonInputLoader(self.input_path)
        self.detector = DataTypeDetector()
        self.normalizer = Normalizer()
        self.decision_engine = IndicatorDecisionEngine()
        self.math_engine = ProcessingMathEngine()
        self.pattern_engine = PatternEngine()
        self.classifier = OutputClassifier()
        self.output_builder = OutputBuilder(max_rows=max_rows)

    def run(self) -> dict[str, Any]:
        records = self.loader.load()
        blocks = [self._process_record(record.source_name, record.payload) for record in records]
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
            payload["metadata_outputs"] = [
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

    def _process_record(self, source_name: str, payload: dict[str, Any]) -> dict[str, Any]:
        detected = self.detector.detect(payload, source_name=source_name)
        normalized = self.normalizer.normalize(payload, detected)
        decision = self.decision_engine.decide(detected, normalized)
        math_result = self.math_engine.compute(normalized, decision)
        patterns = self.pattern_engine.detect(normalized, math_result, decision)
        routes = self.classifier.classify(detected, normalized, math_result, patterns, decision)

        return {
            "source_name": source_name,
            "detected": detected,
            "normalized": normalized,
            "decision": decision,
            "math": math_result,
            "patterns": patterns,
            "routes": routes,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Processing-Signals main pipeline: input -> processing -> math -> classification -> output."
    )
    parser.add_argument(
        "--input",
        default=None,
        help="Input JSON file, ZIP file, or directory containing JSON files. Defaults to the first supported file in data_input/.",
    )
    parser.add_argument(
        "--output",
        default="data_output/main_pipeline_output.json",
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
        help="Write data_output/validation_report.json in addition to embedding validation in the main output.",
    )
    parser.add_argument(
        "--write-manifest",
        action="store_true",
        help="Write data_output/metadata/manifest.json in addition to embedding manifest in the main output.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = resolve_input_path(args.input)
    pipeline = MainPipeline(
        input_path=input_path,
        output_path=Path(args.output),
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
    print(Path(args.output).resolve())
    print()
    print(f"validation_status: {result.get('validation_status')}")
    print()
    if args.write_validation_report:
        print("validation_report:")
        print(Path(args.output).parent / "validation_report.json")
        print()
    if args.write_manifest:
        print("metadata_outputs:")
        for output in result.get("metadata_outputs", []):
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


if __name__ == "__main__":
    main()
