from __future__ import annotations

import numpy as np
import pandas as pd

from signal_analysis.core.schema import validate_dataframe
from signal_analysis.indicators.states import build_signal_state_frame

# \file **********************************************************************
# COMPANY:            ELATIN
# PROJECT:            SIGNAL_ANALYSIS
# COMPONENT:          CORE
# MODULE NAME:        window_sections.py
# DESCRIPTION:        @brief Shared section builders used by WindowBuilder.
# CREATION DATE:      06.05.2026
# VERSION:            $Revision: 0.1$
# CHANGES:            06.05.2026 - Extracted reusable section helper functions.
# *****************************************************************************


class WindowSectionBuilders:
    # *******************************************************************************************************************
    # Functionname:       WindowSectionBuilders._summarize_frame(df, columns, prefix, include)
    #
    # @brief              Convert a DataFrame section into compact scalar features.
    # @pre                df is a DataFrame (empty allowed).
    # @post               Returns flat dict of scalar stats; input is not mutated.
    # @param[in]          df: Input DataFrame section
    #                     columns: Optional columns to summarize
    #                     prefix: Optional key prefix
    #                     include: Stats to compute
    # @param[out]         out: Flat dict {prefix}{column}_{stat}: float
    #
    # @callsequence       @startuml
    #                     title WindowSectionBuilders._summarize_frame
    #                     start
    #                     :validate_dataframe(df);
    #                     if (df.empty?) then (yes)
    #                       :return empty dict;
    #                       stop
    #                     endif
    #                     :select numeric columns;
    #                     :resolve target columns;
    #                     repeat
    #                       :compute requested scalar statistics;
    #                       :append flattened output keys;
    #                     repeat while (more columns?)
    #                     :return summary dict;
    #                     end
    #                     @enduml
    # *******************************************************************************************************************
    @staticmethod
    def _summarize_frame(
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
            target_cols = [col for col in columns if col in numeric_df.columns]

        for col in target_cols:
            series = numeric_df[col]
            if series.empty:
                continue

            first_val = series.iloc[0]
            last_val = series.iloc[-1]

            if "mean" in include_set:
                out[f"{prefix}{col}_mean"] = float(series.mean())
            if "std" in include_set:
                out[f"{prefix}{col}_std"] = float(series.std())
            if "min" in include_set:
                out[f"{prefix}{col}_min"] = float(series.min())
            if "max" in include_set:
                out[f"{prefix}{col}_max"] = float(series.max())
            if "first" in include_set:
                out[f"{prefix}{col}_first"] = float(first_val)
            if "last" in include_set:
                out[f"{prefix}{col}_last"] = float(last_val)
            if "delta" in include_set:
                out[f"{prefix}{col}_delta"] = float(last_val - first_val)
            if "median" in include_set:
                out[f"{prefix}{col}_median"] = float(series.median())

        return out

    # *******************************************************************************************************************
    # Functionname:       WindowSectionBuilders.detect_existing_columns(df, candidates)
    #
    # @brief              Return candidate columns that exist in df, preserving order.
    # @pre                df is a DataFrame.
    # @post               Returns ordered subset of candidates present in df.columns.
    # @param[in]          df: Input DataFrame
    #                     candidates: Ordered list of candidate column names
    # @param[out]         out: Ordered list of existing column names
    #
    # @callsequence       @startuml
    #                     title WindowSectionBuilders.detect_existing_columns
    #                     start
    #                     :iterate candidates in input order;
    #                     :keep only candidates present in df.columns;
    #                     :return filtered ordered list;
    #                     end
    #                     @enduml
    # *******************************************************************************************************************
    @staticmethod
    def detect_existing_columns(
        df: pd.DataFrame,
        candidates: list[str],
    ) -> list[str]:
        return [col for col in candidates if col in df.columns]

    # *******************************************************************************************************************
    # Functionname:       WindowSectionBuilders.build_volume_flow_section(df)
    #
    # @brief              Build compact summary for volume/activity/flow-like columns.
    # @pre                df is a DataFrame.
    # @post               Returns summary dict or None if no compatible columns exist.
    # @param[in]          df: Input window DataFrame
    # @param[out]         out: Flat summary dict for volume_flow section, or None
    #
    # @callsequence       @startuml
    #                     title WindowSectionBuilders.build_volume_flow_section
    #                     start
    #                     :define supported volume/flow candidates;
    #                     :detect_existing_columns(df, candidates);
    #                     if (no compatible columns?) then (yes)
    #                       :return None;
    #                       stop
    #                     endif
    #                     :_summarize_frame(df, cols, prefix="");
    #                     if (summary empty?) then (yes)
    #                       :return None;
    #                     else (no)
    #                       :return summary;
    #                     endif
    #                     end
    #                     @enduml
    # *******************************************************************************************************************
    @staticmethod
    def build_volume_flow_section(df: pd.DataFrame) -> dict | None:
        candidates = [
            "volume",
            "activity",
            "energy",
            "positive_flow",
            "negative_flow",
            "flow_delta",
            "flow_imbalance",
            "cumulative_flow",
            "buy_volume",
            "sell_volume",
            "volume_delta",
            "volume_imbalance",
            "cvd",
            "obv",
            "vwap",
            "rvol",
            "mfi",
            "cmf",
        ]
        cols = WindowSectionBuilders.detect_existing_columns(df, candidates)
        if not cols:
            return None
        section = WindowSectionBuilders._summarize_frame(df, columns=cols, prefix="")
        return section if section else None

    # *******************************************************************************************************************
    # Functionname:       WindowSectionBuilders.build_technical_indicators_section(df)
    #
    # @brief              Build compact summary for pre-existing technical indicator columns.
    # @pre                df is a DataFrame.
    # @post               Returns summary dict or None if no compatible columns exist.
    # @param[in]          df: Input window DataFrame
    # @param[out]         out: Flat summary dict for technical_indicators section, or None
    #
    # @callsequence       @startuml
    #                     title WindowSectionBuilders.build_technical_indicators_section
    #                     start
    #                     :define supported indicator candidates;
    #                     :detect_existing_columns(df, candidates);
    #                     if (no compatible columns?) then (yes)
    #                       :return None;
    #                       stop
    #                     endif
    #                     :_summarize_frame(df, cols, prefix="");
    #                     if (summary empty?) then (yes)
    #                       :return None;
    #                     else (no)
    #                       :return summary;
    #                     endif
    #                     end
    #                     @enduml
    # *******************************************************************************************************************
    @staticmethod
    def build_technical_indicators_section(df: pd.DataFrame) -> dict | None:
        candidates = [
            "rsi",
            "tsi",
            "roc",
            "macd",
            "macd_signal",
            "macd_histogram",
            "atr",
            "adx",
            "di_plus",
            "di_minus",
            "bb_upper",
            "bb_mid",
            "bb_lower",
            "bb_width",
            "stoch_k",
            "stoch_d",
            "williams_r",
            "stoch_rsi",
            "ma_fast",
            "ma_slow",
            "ma_distance",
            "sma",
            "ema",
            "wma",
            "kama",
        ]
        cols = WindowSectionBuilders.detect_existing_columns(df, candidates)
        if not cols:
            return None
        section = WindowSectionBuilders._summarize_frame(df, columns=cols, prefix="")
        return section if section else None

    # *******************************************************************************************************************
    # Functionname:       WindowSectionBuilders.build_trend_signals_section(df)
    #
    # @brief              Build compact trend-signals section from domain-agnostic state indicators.
    # @pre                df is a DataFrame.
    # @post               Returns summarized trend_signals dict or None.
    # @param[in]          df: Input window DataFrame
    # @param[out]         out: Flat trend_signals summary dict, or None
    #
    # @callsequence       @startuml
    #                     title WindowSectionBuilders.build_trend_signals_section
    #                     start
    #                     if (df.empty?) then (yes)
    #                       :return None;
    #                       stop
    #                     endif
    #                     :build_signal_state_frame(df);
    #                     if (state frame build fails or is empty?) then (yes)
    #                       :return None;
    #                       stop
    #                     endif
    #                     :collect *_state last and mode values;
    #                     :summarize *_confidence numeric columns;
    #                     if (output empty?) then (yes)
    #                       :return None;
    #                     else (no)
    #                       :return summary;
    #                     endif
    #                     end
    #                     @enduml
    # *******************************************************************************************************************
    @staticmethod
    def build_trend_signals_section(df: pd.DataFrame) -> dict | None:
        if df.empty:
            return None

        try:
            state_df = build_signal_state_frame(df)
        except Exception:
            return None

        if state_df.empty:
            return None

        out: dict[str, float | str] = {}
        state_cols = [col for col in state_df.columns if col.endswith("_state")]
        conf_cols = [col for col in state_df.columns if col.endswith("_confidence")]

        for col in state_cols:
            series = state_df[col].dropna()
            if series.empty:
                continue
            out[f"{col}_last"] = str(series.iloc[-1])
            mode_vals = series.mode(dropna=True)
            if not mode_vals.empty:
                out[f"{col}_mode"] = str(mode_vals.iloc[0])

        if conf_cols:
            conf_summary = WindowSectionBuilders._summarize_frame(
                state_df,
                columns=conf_cols,
                prefix="",
            )
            out.update(conf_summary)

        return out if out else None


def detect_existing_columns(
    df: pd.DataFrame,
    candidates: list[str],
) -> list[str]:
    return WindowSectionBuilders.detect_existing_columns(df, candidates)


def build_volume_flow_section(df: pd.DataFrame) -> dict | None:
    return WindowSectionBuilders.build_volume_flow_section(df)


def build_technical_indicators_section(df: pd.DataFrame) -> dict | None:
    return WindowSectionBuilders.build_technical_indicators_section(df)


def build_trend_signals_section(df: pd.DataFrame) -> dict | None:
    return WindowSectionBuilders.build_trend_signals_section(df)


__all__ = [
    "WindowSectionBuilders",
    "detect_existing_columns",
    "build_volume_flow_section",
    "build_technical_indicators_section",
    "build_trend_signals_section",
]