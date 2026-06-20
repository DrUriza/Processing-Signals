# \file **********************************************************************
# COMPANY:            ELATIN
# PROJECT:            SIGNAL_ANALYSIS
# COMPONENT:          CORE
# MODULE NAME:        feature_window.py
# DESCRIPTION:        @brief Domain-agnostic temporal feature window schema.
#                     Defines the canonical per-window data contract for
#                     multi-layer signal analysis pipelines.
# CREATION DATE:      06.05.2026
# VERSION:            $Revision: 0.1$
# CHANGES:            06.05.2026 - Initial implementation.
# *****************************************************************************

from __future__ import annotations

import pandas as pd

from signal_analysis.core.schema import (
    OPTIONAL_TIME_COLUMNS,
    REQUIRED_OHLC_COLUMNS,
    infer_time_column,
    validate_dataframe,
    validate_required_columns,
)

# *********************************************************************************************************************
# Window Section Keys
# *********************************************************************************************************************

#: Canonical top-level section keys for a feature window.
WINDOW_SECTION_KEYS = [
    "window_meta",
    "variable_action",
    "volume_flow",
    "technical_indicators",
    "trend_signals",
    "fourier_features",
    "wavelet_features",
    "tda_features",
    "invariant_features",
    "dynamics_features",
    "label_or_future_state",
]

#: Minimum required metadata keys inside window_meta.
WINDOW_META_KEYS = [
    "window_id",
    "n_samples",
    "schema_version",
]


# *********************************************************************************************************************
# Functionname:       build_window_meta(window_id, n_samples, schema_version, extra)
#
# @brief              Build window metadata section.
# @pre                window_id is hashable; n_samples > 0.
# @post               Returns dict with standard metadata keys.
# @param[in]          window_id: Unique identifier for the window
#                     n_samples: Number of samples in the window
#                     schema_version: Version string of the feature schema
#                     extra: Additional metadata (optional)
# @param[out]         out: Window metadata dict
# *********************************************************************************************************************
def build_window_meta(
    window_id: int | str,
    n_samples: int,
    schema_version: str = "1.0",
    extra: dict | None = None,
) -> dict:
    meta = {
        "window_id": window_id,
        "n_samples": n_samples,
        "schema_version": schema_version,
    }
    if extra:
        meta.update(extra)
    return meta


# *********************************************************************************************************************
# Functionname:       build_empty_window()
#
# @brief              Build an empty feature window with all section keys set to None.
# @pre                None
# @post               Returns dict with all WINDOW_SECTION_KEYS set to None.
# @param[out]         out: Empty feature window dict
# *********************************************************************************************************************
def build_empty_window() -> dict:
    return {key: None for key in WINDOW_SECTION_KEYS}


# *********************************************************************************************************************
# Functionname:       extract_window_from_dataframe(df, start, end, window_id, schema_version)
#
# @brief              Extract a feature window slice from a DataFrame.
# @pre                df is a non-empty DataFrame with OHLC columns.
# @post               Returns a feature window dict with variable_action and window_meta populated.
# @param[in]          df: Source DataFrame (must contain OHLC columns)
#                     start: Start index position (inclusive, integer positional)
#                     end: End index position (exclusive, integer positional)
#                     window_id: Optional window identifier; uses start if None
#                     schema_version: Version string of the feature schema
# @param[out]         out: Populated feature window dict
#
# @callsequence       @startuml
#                     title extract_window_from_dataframe
#                     start
#                     :validate_dataframe(df);
#                     :validate OHLC columns exist;
#                     :Slice df[start:end];
#                     :Build window_meta from slice shape;
#                     :Populate variable_action with OHLC slice;
#                     :Infer and attach time info if present;
#                     :Return populated window dict;
#                     end
#                     @enduml
# *********************************************************************************************************************
def extract_window_from_dataframe(
    df: pd.DataFrame,
    start: int,
    end: int,
    window_id: int | str | None = None,
    schema_version: str = "1.0",
) -> dict:
    validate_dataframe(df)
    validate_required_columns(df, REQUIRED_OHLC_COLUMNS)

    if window_id is None:
        window_id = start

    slice_df = df.iloc[start:end]

    time_col = infer_time_column(df)
    time_start = None
    time_end = None
    if time_col is not None and not slice_df.empty:
        time_start = slice_df[time_col].iloc[0] if len(slice_df) > 0 else None
        time_end = slice_df[time_col].iloc[-1] if len(slice_df) > 0 else None

    meta = build_window_meta(
        window_id=window_id,
        n_samples=len(slice_df),
        schema_version=schema_version,
        extra={
            "time_start": time_start,
            "time_end": time_end,
            "time_column": time_col,
        },
    )

    window = build_empty_window()
    window["window_meta"] = meta
    window["variable_action"] = slice_df[REQUIRED_OHLC_COLUMNS].copy()

    return window


# *********************************************************************************************************************
# Functionname:       build_windowed_sequence(df, window_size, step, schema_version)
#
# @brief              Generate a sequence of feature windows from a DataFrame.
# @pre                df has OHLC columns; window_size > 0; step > 0.
# @post               Returns list of feature window dicts.
# @param[in]          df: Source DataFrame
#                     window_size: Number of samples per window
#                     step: Step between windows
#                     schema_version: Version string for all windows
# @param[out]         out: List of feature window dicts
# *********************************************************************************************************************
def build_windowed_sequence(
    df: pd.DataFrame,
    window_size: int,
    step: int | None = None,
    schema_version: str = "1.0",
) -> list[dict]:
    validate_dataframe(df)
    validate_required_columns(df, REQUIRED_OHLC_COLUMNS)

    if window_size < 1:
        raise ValueError("window_size must be >= 1")
    if step is None:
        step = window_size
    if step < 1:
        raise ValueError("step must be >= 1")

    n = len(df)
    windows = []
    window_id = 0

    for start in range(0, n - window_size + 1, step):
        end = start + window_size
        w = extract_window_from_dataframe(
            df=df,
            start=start,
            end=end,
            window_id=window_id,
            schema_version=schema_version,
        )
        windows.append(w)
        window_id += 1

    return windows


# *********************************************************************************************************************
# Functionname:       validate_window(window)
#
# @brief              Validate that a feature window dict has required structure.
# @pre                window should be a dict.
# @post               Raises ValueError if validation fails; returns None on success.
# @param[in]          window: Feature window dict to validate
# @param[out]         None
# *********************************************************************************************************************
def validate_window(window: dict) -> None:
    if not isinstance(window, dict):
        raise ValueError("Feature window must be a dict.")
    missing = set(WINDOW_SECTION_KEYS) - set(window.keys())
    if missing:
        raise ValueError(
            f"Feature window is missing sections: {sorted(missing)}"
        )


__all__ = [
    "WINDOW_SECTION_KEYS",
    "WINDOW_META_KEYS",
    "build_window_meta",
    "build_empty_window",
    "extract_window_from_dataframe",
    "build_windowed_sequence",
    "validate_window",
]
