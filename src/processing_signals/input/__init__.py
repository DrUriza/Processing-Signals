"""Input package exports.

Some input pipeline modules are optional in the current repository snapshot.
Keep package import lightweight so API registries can be imported on their own.
"""

from __future__ import annotations

from typing import Any


__all__ = ["RuntimeConfig", "InputPipeline", "load_runtime_config"]


def __getattr__(name: str) -> Any:
    if name in {"RuntimeConfig", "load_runtime_config"}:
        from processing_signals.input.config import RuntimeConfig, load_runtime_config

        return {"RuntimeConfig": RuntimeConfig, "load_runtime_config": load_runtime_config}[name]
    if name == "InputPipeline":
        from processing_signals.input.input_pipeline import InputPipeline

        return InputPipeline
    raise AttributeError(name)

