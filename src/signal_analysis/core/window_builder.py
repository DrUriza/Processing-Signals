# \file **********************************************************************
# COMPANY:            ELATIN
# PROJECT:            SIGNAL_ANALYSIS
# COMPONENT:          CORE
# MODULE NAME:        window_builder.py
# DESCRIPTION:        @brief Domain-agnostic feature window builder.
#                     Orchestrates per-window feature sections and creates
#                     flat tabular representations for downstream ML tasks.
# CREATION DATE:      06.05.2026
# VERSION:            $Revision: 0.1$
# CHANGES:            06.05.2026 - Initial implementation.
# *****************************************************************************

from __future__ import annotations

import numpy as np
import pandas as pd

from signal_analysis.core.feature_window import build_empty_window, build_window_meta, validate_window
from signal_analysis.core.schema import REQUIRED_OHLC_COLUMNS, infer_time_column, validate_dataframe
from signal_analysis.core.window_sections import (
    build_technical_indicators_section,
    build_trend_signals_section,
    build_volume_flow_section,
)
from signal_analysis.indicators.dynamics import add_dynamics_features
from signal_analysis.indicators.invariants import add_invariant_features
from signal_analysis.indicators.variable_action import add_ohlc_geometry_features


class WindowBuilder:

    # ***********************************************************************************************************************
    # Functionname:       WindowBuilder.summarize_frame(df, columns, prefix, include)
    #
    # @brief              Convert a DataFrame section into compact scalar features.
    # @pre                df is a DataFrame (empty allowed).
    # @post               Returns flat dict of scalar stats; input is not mutated.
    # @param[in]          df: Input DataFrame section
    #                     columns: Optional columns to summarize (None => all numeric columns)
    #                     prefix: Optional key prefix
    #                     include: Stats to compute
    # @param[out]         out: Flat dict {prefix}{column}_{stat}: float
    #
    # @callsequence       @startuml
    #                     title summarize_frame
    #                     start
    #                     :select target columns;
    #                     :filter missing and non-numeric;
    #                     repeat
    #                       :compute requested stats;
    #                       :write scalar keys to output dict;
    #                     repeat while (more columns?)
    #                     :return summary dict;
    #                     end
    #                     @enduml
    # ***********************************************************************************************************************
    @staticmethod
    def summarize_frame(
        df: pd.DataFrame,
        columns: list[str] | None = None,
        prefix: str = "",
        include: tuple[str, ...] = (
            "mean",
            "std",
            "min",
            "max",
            "first",
            "last",
            "delta",
            "median",
        ),
    ) -> dict[str, float]:
        validate_dataframe(df)

        out: dict[str, float] = {}
        if df.empty:
            return out

        include_set = set(include)
        numeric_df = df.select_dtypes(include=[np.number])

        if columns is None:
            target_cols = list(numeric_df.columns)
        else:
            target_cols = [c for c in columns if c in numeric_df.columns]

        for col in target_cols:
            s = numeric_df[col]
            if s.empty:
                continue

            first_val = s.iloc[0]
            last_val = s.iloc[-1]

            if "mean" in include_set:
                out[f"{prefix}{col}_mean"] = float(s.mean())
            if "std" in include_set:
                out[f"{prefix}{col}_std"] = float(s.std())
            if "min" in include_set:
                out[f"{prefix}{col}_min"] = float(s.min())
            if "max" in include_set:
                out[f"{prefix}{col}_max"] = float(s.max())
            if "first" in include_set:
                out[f"{prefix}{col}_first"] = float(first_val)
            if "last" in include_set:
                out[f"{prefix}{col}_last"] = float(last_val)
            if "delta" in include_set:
                out[f"{prefix}{col}_delta"] = float(last_val - first_val)
            if "median" in include_set:
                out[f"{prefix}{col}_median"] = float(s.median())

        return out

    # ***********************************************************************************************************************
    # Functionname:       WindowBuilder.build_feature_sections(df, value_columns, invariant_window,
    #                              dynamics_periods)
    #
    # @brief              Compute available feature sections from a DataFrame window.
    # @pre                df is a DataFrame; value_columns optional.
    # @post               Returns dict with canonical section keys for available computations.
    # @param[in]          df: Input DataFrame window
    #                     value_columns: Optional columns for invariant/dynamics computation
    #                     invariant_window: Rolling window for invariant features
    #                     dynamics_periods: Finite-difference periods for dynamics features
    # @param[out]         out: Dict of feature sections
    #
    # @callsequence       @startuml
    #                     title build_feature_sections
    #                     start
    #                     :validate_dataframe(df);
    #                     if (OHLC columns available?) then (yes)
    #                       :compute add_ohlc_geometry_features(df);
    #                       :summarize variable-action numeric features;
    #                     endif
    #                     :resolve value columns;
    #                     if (value columns exist?) then (yes)
    #                       :compute invariant features and summarize;
    #                       :compute dynamics features and summarize;
    #                     endif
    #                     :fill unavailable sections as None;
    #                     :return section dict;
    #                     end
    #                     @enduml
    # ***********************************************************************************************************************
    @staticmethod
    def build_feature_sections(
        df: pd.DataFrame,
        value_columns: list[str] | None = None,
        invariant_window: int = 20,
        dynamics_periods: int = 1,
    ) -> dict:
        validate_dataframe(df)

        sections = {
            "variable_action": None,
            "volume_flow": None,
            "invariant_features": None,
            "dynamics_features": None,
            "technical_indicators": None,
            "trend_signals": None,
            "fourier_features": None,
            "wavelet_features": None,
            "tda_features": None,
        }

        sections["volume_flow"] = build_volume_flow_section(df)
        sections["technical_indicators"] = build_technical_indicators_section(df)
        sections["trend_signals"] = build_trend_signals_section(df)

        has_ohlc = all(col in df.columns for col in REQUIRED_OHLC_COLUMNS)
        if has_ohlc and not df.empty:
            ohlc_enriched = add_ohlc_geometry_features(df)
            new_cols = [c for c in ohlc_enriched.columns if c not in df.columns]
            variable_action = WindowBuilder.summarize_frame(
                ohlc_enriched,
                columns=new_cols,
                prefix="",
            )
            sections["variable_action"] = variable_action if variable_action else None

        if value_columns is None:
            value_candidates = list(df.select_dtypes(include=[np.number]).columns)
        else:
            value_candidates = list(value_columns)

        valid_value_columns = [
            col
            for col in value_candidates
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col])
        ]

        if valid_value_columns and not df.empty:
            invariant_df = add_invariant_features(
                df,
                columns=valid_value_columns,
                window=invariant_window,
            )
            invariant_cols = [c for c in invariant_df.columns if c not in df.columns]
            invariant_features = WindowBuilder.summarize_frame(
                invariant_df,
                columns=invariant_cols,
                prefix="",
            )
            sections["invariant_features"] = (
                invariant_features if invariant_features else None
            )

            dynamics_df = add_dynamics_features(
                df,
                columns=valid_value_columns,
                periods=dynamics_periods,
            )
            dynamics_cols = [c for c in dynamics_df.columns if c not in df.columns]
            dynamics_features = WindowBuilder.summarize_frame(
                dynamics_df,
                columns=dynamics_cols,
                prefix="",
            )
            sections["dynamics_features"] = dynamics_features if dynamics_features else None

        return sections

    # ***********************************************************************************************************************
    # Functionname:       WindowBuilder.build_window(df, start, end, window_id, schema_version,
    #                              value_columns)
    #
    # @brief              Build one canonical feature window.
    # @pre                df is a DataFrame; start/end define a valid positional slice.
    # @post               Returns validated canonical window dict with available sections populated.
    # @param[in]          df: Source DataFrame
    #                     start: Start row index (inclusive, positional)
    #                     end: End row index (exclusive, positional)
    #                     window_id: Optional window identifier
    #                     schema_version: Schema version string
    #                     value_columns: Optional columns for invariant/dynamics features
    # @param[out]         out: Canonical validated feature window dict
    #
    # @callsequence       @startuml
    #                     title build_window
    #                     start
    #                     :validate_dataframe(df);
    #                     :slice window_df = df.iloc[start:end];
    #                     :build_empty_window();
    #                     :build metadata with build_window_meta(...);
    #                     :build_feature_sections(window_df,...);
    #                     :assign canonical sections;
    #                     :validate_window(window);
    #                     :return window;
    #                     end
    #                     @enduml
    # ***********************************************************************************************************************
    @staticmethod
    def build_window(
        df: pd.DataFrame,
        start: int,
        end: int,
        window_id: str | None = None,
        schema_version: str = "1.0",
        value_columns: list[str] | None = None,
    ) -> dict:
        validate_dataframe(df)

        window_df = df.iloc[start:end].copy()
        if window_id is None:
            window_id = str(start)

        time_col = infer_time_column(df)
        time_start = None
        time_end = None
        if time_col is not None and not window_df.empty:
            time_start = window_df[time_col].iloc[0]
            time_end = window_df[time_col].iloc[-1]

        meta = build_window_meta(
            window_id=window_id,
            n_samples=len(window_df),
            schema_version=schema_version,
            extra={
                "start": start,
                "end": end,
                "time_column": time_col,
                "time_start": time_start,
                "time_end": time_end,
            },
        )

        window = build_empty_window()
        window["window_meta"] = meta

        sections = WindowBuilder.build_feature_sections(
            window_df,
            value_columns=value_columns,
        )

        window["variable_action"] = sections["variable_action"]
        window["volume_flow"] = sections["volume_flow"]
        window["technical_indicators"] = sections["technical_indicators"]
        window["trend_signals"] = sections["trend_signals"]
        window["fourier_features"] = sections["fourier_features"]
        window["wavelet_features"] = sections["wavelet_features"]
        window["tda_features"] = sections["tda_features"]
        window["invariant_features"] = sections["invariant_features"]
        window["dynamics_features"] = sections["dynamics_features"]
        window["label_or_future_state"] = None

        validate_window(window)
        return window

    # ***********************************************************************************************************************
    # Functionname:       WindowBuilder.build_window_sequence(df, window_size, step, value_columns,
    #                              schema_version)
    #
    # @brief              Build a sequence of canonical feature windows from a DataFrame.
    # @pre                df is a DataFrame; window_size >= 1; step >= 1.
    # @post               Returns ordered list of validated windows.
    # @param[in]          df: Source DataFrame
    #                     window_size: Number of rows per window
    #                     step: Step between consecutive windows
    #                     value_columns: Optional columns for invariant/dynamics features
    #                     schema_version: Schema version string
    # @param[out]         out: Ordered list of window dicts
    #
    # @callsequence       @startuml
    #                     title build_window_sequence
    #                     start
    #                     :validate_dataframe(df);
    #                     :validate window_size and step;
    #                     repeat
    #                       :build_window(df, start, end, ...);
    #                       :append to windows list;
    #                     repeat while (more windows?)
    #                     :return windows list;
    #                     end
    #                     @enduml
    # ***********************************************************************************************************************
    @staticmethod
    def build_window_sequence(
        df: pd.DataFrame,
        window_size: int,
        step: int,
        value_columns: list[str] | None = None,
        schema_version: str = "1.0",
    ) -> list[dict]:
        validate_dataframe(df)

        if window_size < 1:
            raise ValueError("window_size must be >= 1")
        if step < 1:
            raise ValueError("step must be >= 1")

        windows: list[dict] = []
        for start in range(0, len(df) - window_size + 1, step):
            end = start + window_size
            window = WindowBuilder.build_window(
                df,
                start=start,
                end=end,
                window_id=str(len(windows)),
                schema_version=schema_version,
                value_columns=value_columns,
            )
            windows.append(window)

        return windows

    # ***********************************************************************************************************************
    # Functionname:       WindowBuilder.flatten_window(window, include_meta, separator)
    #
    # @brief              Flatten one nested canonical window into scalar key-value pairs.
    # @pre                window is a canonical window dict.
    # @post               Returns flat dict; skips None sections and non-scalar values.
    # @param[in]          window: Canonical feature window dict
    #                     include_meta: Include window_meta section if True
    #                     separator: Key separator for flattened keys
    # @param[out]         out: Flat scalar dictionary
    #
    # @callsequence       @startuml
    #                     title flatten_window
    #                     start
    #                     :validate_window(window);
    #                     :iterate sections recursively;
    #                     if (value is scalar?) then (yes)
    #                       :store flattened key;
    #                     else (no)
    #                       :skip non-scalar values;
    #                     endif
    #                     :return flat dict;
    #                     end
    #                     @enduml
    # ***********************************************************************************************************************
    @staticmethod
    def flatten_window(
        window: dict,
        include_meta: bool = False,
        separator: str = "__",
    ) -> dict[str, float | int | str]:
        validate_window(window)

        flat: dict[str, float | int | str] = {}

        def _is_scalar(value: object) -> bool:
            return isinstance(value, (str, int, float, bool, np.integer, np.floating, np.bool_))

        def _walk(prefix: str, obj: object) -> None:
            if obj is None:
                return
            if isinstance(obj, dict):
                for k, v in obj.items():
                    next_prefix = f"{prefix}{separator}{k}" if prefix else str(k)
                    _walk(next_prefix, v)
                return
            if _is_scalar(obj):
                if isinstance(obj, (np.integer, np.floating, np.bool_)):
                    flat[prefix] = obj.item()
                else:
                    flat[prefix] = obj

        for section_key, section_value in window.items():
            if section_key == "window_meta" and not include_meta:
                continue
            if section_value is None:
                continue
            _walk(section_key, section_value)

        return flat

    # ***********************************************************************************************************************
    # Functionname:       WindowBuilder.windows_to_feature_frame(windows, include_meta)
    #
    # @brief              Convert a list of canonical windows to a tabular feature DataFrame.
    # @pre                windows is a list of canonical window dicts.
    # @post               Returns DataFrame with one row per window, preserving order.
    # @param[in]          windows: List of canonical windows
    #                     include_meta: Include metadata fields if True
    # @param[out]         out: Feature DataFrame
    #
    # @callsequence       @startuml
    #                     title windows_to_feature_frame
    #                     start
    #                     repeat
    #                       :flatten_window(window, include_meta);
    #                       :append row dict;
    #                     repeat while (more windows?)
    #                     :pd.DataFrame(rows);
    #                     :return DataFrame;
    #                     end
    #                     @enduml
    # ***********************************************************************************************************************
    @staticmethod
    def windows_to_feature_frame(
        windows: list[dict],
        include_meta: bool = False,
    ) -> pd.DataFrame:
        rows: list[dict[str, float | int | str]] = []
        for window in windows:
            rows.append(
                WindowBuilder.flatten_window(
                    window,
                    include_meta=include_meta,
                )
            )
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Module-level convenience wrappers
# ---------------------------------------------------------------------------


def summarize_frame(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    prefix: str = "",
    include: tuple[str, ...] = (
        "mean",
        "std",
        "min",
        "max",
        "first",
        "last",
        "delta",
        "median",
    ),
) -> dict[str, float]:
    return WindowBuilder.summarize_frame(df, columns=columns, prefix=prefix, include=include)


def build_feature_sections(
    df: pd.DataFrame,
    value_columns: list[str] | None = None,
    invariant_window: int = 20,
    dynamics_periods: int = 1,
) -> dict:
    return WindowBuilder.build_feature_sections(
        df,
        value_columns=value_columns,
        invariant_window=invariant_window,
        dynamics_periods=dynamics_periods,
    )


def build_window(
    df: pd.DataFrame,
    start: int,
    end: int,
    window_id: str | None = None,
    schema_version: str = "1.0",
    value_columns: list[str] | None = None,
) -> dict:
    return WindowBuilder.build_window(
        df,
        start=start,
        end=end,
        window_id=window_id,
        schema_version=schema_version,
        value_columns=value_columns,
    )


def build_window_sequence(
    df: pd.DataFrame,
    window_size: int,
    step: int,
    value_columns: list[str] | None = None,
    schema_version: str = "1.0",
) -> list[dict]:
    return WindowBuilder.build_window_sequence(
        df,
        window_size=window_size,
        step=step,
        value_columns=value_columns,
        schema_version=schema_version,
    )


def flatten_window(
    window: dict,
    include_meta: bool = False,
    separator: str = "__",
) -> dict[str, float | int | str]:
    return WindowBuilder.flatten_window(
        window,
        include_meta=include_meta,
        separator=separator,
    )


def windows_to_feature_frame(
    windows: list[dict],
    include_meta: bool = False,
) -> pd.DataFrame:
    return WindowBuilder.windows_to_feature_frame(windows, include_meta=include_meta)


__all__ = [
    "WindowBuilder",
    "summarize_frame",
    "build_feature_sections",
    "build_window",
    "build_window_sequence",
    "flatten_window",
    "windows_to_feature_frame",
]
