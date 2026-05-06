# \file **********************************************************************
# COMPANY:            ELATIN
# PROJECT:            SIGNAL_ANALYSIS
# COMPONENT:          CORE
# MODULE NAME:        schema.py
# DESCRIPTION:        @brief Domain-agnostic signal data schema and validation contract
# CREATION DATE:      06.05.2026
# VERSION:            $Revision: 0.1$
# CHANGES:            06.05.2026 - Initial schema constants and validation functions.
# *****************************************************************************

from __future__ import annotations

import pandas as pd

# *********************************************************************************************************************
# Schema Constants
# *********************************************************************************************************************

#: Canonical OHLC columns required for temporal signal analysis.
#: Agnostic to domain (trading, robotics, fluids, radar, mathematics).
REQUIRED_OHLC_COLUMNS = ["open", "high", "low", "close"]

#: Optional volume/magnitude column names.
OPTIONAL_VOLUME_COLUMNS = ["volume"]

#: Optional signed flow columns (e.g., buyer/seller pressure or directional magnitudes).
#: Agnostic: can represent pressure, flux, gradient, or other signed quantities.
OPTIONAL_SIGNED_FLOW_COLUMNS = [
    "positive_flow",
    "negative_flow",
]

#: Optional time column names in priority order.
#: First match is returned by infer_time_column().
OPTIONAL_TIME_COLUMNS = [
    "timestamp",
    "time",
    "datetime",
]

#: Data quality and coverage metadata columns.
QUALITY_COLUMNS = [
    "coverage_ratio",
    "missing_ratio",
    "quality_flag",
]


# *********************************************************************************************************************
# Functionname:       validate_dataframe(df: pd.DataFrame)
#
# @brief              Validate input is a valid, non-empty pandas DataFrame.
# @pre                df should be a pandas DataFrame or DataFrame-like.
# @post               Raises ValueError if validation fails; returns None on success.
# @param[in]          df: Input to validate
# @param[out]         out: None
#
# @callsequence       @startuml
#                     title validate_dataframe
#                     start
#                     if (not isinstance(df, pd.DataFrame)?) then (yes)
#                       :Raise ValueError - not a DataFrame;
#                       stop
#                     endif
#                     if (df.empty?) then (yes)
#                       :Raise ValueError - empty DataFrame;
#                       stop
#                     endif
#                     end
#                     @enduml
#
# @InOutCorrelation   As described in UML diagram
# @traceability
# *********************************************************************************************************************
def validate_dataframe(df: pd.DataFrame) -> None:
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame.")
    if df.empty:
        raise ValueError("DataFrame is empty.")


# *********************************************************************************************************************
# Functionname:       validate_required_columns(df: pd.DataFrame, required_columns: list[str])
#
# @brief              Validate that all required columns exist in DataFrame.
# @pre                df is a DataFrame; required_columns is a list of strings.
# @post               Raises ValueError listing missing columns; returns None on success.
# @param[in]          df: Input DataFrame
#                     required_columns: Column names that must exist
# @param[out]         out: None
#
# @callsequence       @startuml
#                     title validate_required_columns
#                     start
#                     :Compute missing_cols = required_columns - df.columns;
#                     if (missing_cols not empty?) then (yes)
#                       :Raise ValueError listing missing columns;
#                       stop
#                     endif
#                     end
#                     @enduml
#
# @InOutCorrelation   As described in UML diagram
# @traceability
# *********************************************************************************************************************
def validate_required_columns(df: pd.DataFrame, required_columns: list[str]) -> None:
    missing_cols = set(required_columns) - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"Missing required columns: {sorted(missing_cols)}. "
            f"Available: {sorted(df.columns)}"
        )

    missing_cols = set(required_columns) - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"Missing required columns: {sorted(missing_cols)}. "
            f"Available: {sorted(df.columns)}"
        )


# *********************************************************************************************************************
# Functionname:       validate_numeric_columns(df: pd.DataFrame, columns: list[str])
#
# @brief              Validate that specified columns are numeric where they exist.
# @pre                df is a DataFrame; columns is a list of strings.
# @post               Ignores columns not present; raises ValueError for non-numeric existing columns.
# @param[in]          df: Input DataFrame
#                     columns: Column names to check
# @param[out]         out: None
#
# @callsequence       @startuml
#                     title validate_numeric_columns
#                     start
#                     repeat
#                       :For each column in columns;
#                       if (column not in df.columns?) then (yes)
#                         :Skip column;
#                       else (no)
#                         if (df[column] is numeric?) then (yes)
#                           :Continue;
#                         else (no)
#                           :Add to non_numeric list;
#                         endif
#                       endif
#                     repeat while (more columns)
#                     if (non_numeric list not empty?) then (yes)
#                       :Raise ValueError listing non-numeric columns;
#                       stop
#                     endif
#                     end
#                     @enduml
#
# @InOutCorrelation   As described in UML diagram
# @traceability
# *********************************************************************************************************************
def validate_numeric_columns(df: pd.DataFrame, columns: list[str]) -> None:
    non_numeric = []
    for col in columns:
        if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
            non_numeric.append(col)

    if non_numeric:
        raise ValueError(
            f"Non-numeric columns found: {sorted(non_numeric)}. "
            f"All must be numeric."
        )


# *********************************************************************************************************************
# Functionname:       validate_ohlc_columns(df: pd.DataFrame)
#
# @brief              Validate OHLC columns (open, high, low, close) exist and are numeric.
# @pre                df is a DataFrame.
# @post               Raises ValueError if validation fails; returns None on success.
# @param[in]          df: Input DataFrame
# @param[out]         out: None
#
# @callsequence       @startuml
#                     title validate_ohlc_columns
#                     start
#                     :Call validate_dataframe(df);
#                     :Call validate_required_columns(df, REQUIRED_OHLC_COLUMNS);
#                     :Call validate_numeric_columns(df, REQUIRED_OHLC_COLUMNS);
#                     end
#                     @enduml
#
# @InOutCorrelation   As described in UML diagram
# @traceability
# *********************************************************************************************************************
def validate_ohlc_columns(df: pd.DataFrame) -> None:
    validate_dataframe(df)
    validate_required_columns(df, REQUIRED_OHLC_COLUMNS)
    validate_numeric_columns(df, REQUIRED_OHLC_COLUMNS)


# *********************************************************************************************************************
# Functionname:       infer_time_column(df: pd.DataFrame)
#
# @brief              Return first matching time column name, or None.
# @pre                df is a DataFrame.
# @post               Returns column name string or None.
# @param[in]          df: Input DataFrame
# @param[out]         out: Matched column name or None
#
# @callsequence       @startuml
#                     title infer_time_column
#                     start
#                     repeat
#                       :For each column in OPTIONAL_TIME_COLUMNS;
#                       if (column in df.columns?) then (yes)
#                         :Return column name;
#                         stop
#                       endif
#                     repeat while (more candidates)
#                     :Return None;
#                     end
#                     @enduml
#
# @InOutCorrelation   As described in UML diagram
# @traceability
# *********************************************************************************************************************
def infer_time_column(df: pd.DataFrame) -> str | None:
    for col in OPTIONAL_TIME_COLUMNS:
        if col in df.columns:
            return col
    return None


# *********************************************************************************************************************
# Functionname:       build_feature_contract(required_columns, optional_columns, schema_version)
#
# @brief              Build a feature schema contract dictionary.
# @pre                schema_version is a string; columns are lists of strings.
# @post               Returns dict with schema_version, required_columns, optional_columns.
# @param[in]          required_columns: List of required column names (default: REQUIRED_OHLC_COLUMNS)
#                     optional_columns: List of optional column names (default: None)
#                     schema_version: Version string for the contract (default: "1.0")
# @param[out]         out: Feature contract dictionary
#
# @callsequence       @startuml
#                     title build_feature_contract
#                     start
#                     if (required_columns is None?) then (yes)
#                       :Set required_columns = REQUIRED_OHLC_COLUMNS;
#                     endif
#                     if (optional_columns is None?) then (yes)
#                       :Set optional_columns = [];
#                     endif
#                     :Build contract dict with schema_version, required_columns, optional_columns;
#                     :Return contract;
#                     end
#                     @enduml
#
# @InOutCorrelation   As described in UML diagram
# @traceability
# *********************************************************************************************************************
def build_feature_contract(required_columns: list[str] | None = None, 
                           optional_columns: list[str] | None = None,
                           schema_version: str = "1.0") -> dict:

    if required_columns is None:
        required_columns = REQUIRED_OHLC_COLUMNS.copy()
    if optional_columns is None:
        optional_columns = []

    return {"schema_version": schema_version, "required_columns": required_columns, "optional_columns": optional_columns}


__all__ = [
    "REQUIRED_OHLC_COLUMNS",
    "OPTIONAL_VOLUME_COLUMNS",
    "OPTIONAL_SIGNED_FLOW_COLUMNS",
    "OPTIONAL_TIME_COLUMNS",
    "QUALITY_COLUMNS",
    "validate_dataframe",
    "validate_required_columns",
    "validate_numeric_columns",
    "validate_ohlc_columns",
    "infer_time_column",
    "build_feature_contract",
]
