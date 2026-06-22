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
from processing_signals.processing.data_type_detector import DataTypeDetector
from processing_signals.processing.indicator_decision_engine import IndicatorDecisionEngine
from processing_signals.processing.math.math_engine import ProcessingMathEngine
from processing_signals.processing.normalizer import Normalizer
from processing_signals.processing.patterns.pattern_engine import PatternEngine


class MainPipeline:
    """Orchestrates input, processing, math, patterns, classification, and output."""

    def __init__(self, input_path: Path, output_path: Path, max_rows: int | None = None):
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
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
                block["family_output_path"] = str(self.output_path.parent / "metadata" / family_info["output_filename"])
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
        default="data_input/btc_processing_jsons_v2.zip",
        help="Input JSON file, ZIP file, or directory containing JSON files.",
    )
    parser.add_argument(
        "--output",
        default="data_output/main_pipeline_output.json",
        help="Output JSON path.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=20,
        help="Optional limit for time-series rows included in the final HMI payload.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pipeline = MainPipeline(input_path=Path(args.input), output_path=Path(args.output), max_rows=args.max_rows)
    result = pipeline.run()
    print(f"records_processed: {result['summary']['records_processed']}")
    print()
    print("data_types:")
    for data_type, count in result["summary"]["data_types"].items():
        print(f"{data_type}: {count}")
    print()
    print("main_output:")
    print(Path(args.output).resolve())
    print()
    print("metadata_outputs:")
    for output in result.get("family_outputs", {}).get("metadata", []):
        print(output["path"])
    print()
    print("family_outputs:")
    for family in result.get("family_outputs", {}).get("families", []):
        print(f"{family['family_key']}:")
        for output in family.get("outputs", []):
            print(output["path"])
        print()


if __name__ == "__main__":
    main()
