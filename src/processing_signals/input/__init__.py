"""Input package exports."""

from __future__ import annotations

from typing import Any


__all__ = ["InputPipeline"]


def __getattr__(name: str) -> Any:
    if name == "InputPipeline":
        from processing_signals.input.input_pipeline import InputPipeline

        return InputPipeline
    raise AttributeError(name)

