from __future__ import annotations

import argparse
from pathlib import Path

from signal_analysis.main_pipeline import MainPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Processing-Signals main pipeline: input -> processing -> math -> classification -> output."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input JSON file, ZIP file, or directory containing JSON files.",
    )
    parser.add_argument(
        "--output",
        default="output/main_pipeline_output.json",
        help="Output JSON path.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional limit for time-series rows included in the final HMI payload.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pipeline = MainPipeline(input_path=Path(args.input), output_path=Path(args.output), max_rows=args.max_rows)
    result = pipeline.run()
    print(f"Pipeline OK. Records processed: {result['summary']['records_processed']}")
    print(f"Output written to: {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
