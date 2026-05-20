# \file **********************************************************************
# COMPANY:            ELATIN
# PROJECT:            SIGNAL_ANALYSIS
# COMPONENT:          CORE
# MODULE NAME:        feature_pipeline.py
# DESCRIPTION:        @brief Domain-agnostic end-to-end feature pipeline.
#                     Converts temporal DataFrames into feature matrices by
#                     building windows, flattening sections, cleaning features,
#                     and reporting diagnostics.
# CREATION DATE:      06.05.2026
# VERSION:            $Revision: 0.1$
# CHANGES:            06.05.2026 - Initial implementation.
# *****************************************************************************

from __future__ import annotations

import numpy as np
import pandas as pd

from signal_analysis.core.schema import validate_dataframe
from signal_analysis.core.window_builder import WindowBuilder


class FeaturePipeline:
    # ***********************************************************************************************************************
    # Functionname:       FeaturePipeline.build_feature_matrix(df, window_size, step, value_columns,
    #                              include_meta, drop_non_numeric, fill_method)
    #
    # @brief              Build end-to-end feature matrix from temporal DataFrame windows.
    # @pre                df is a non-empty DataFrame; window_size > 0; step > 0.
    # @post               Returns feature DataFrame; input df is not mutated.
    # @param[in]          df: Input temporal DataFrame
    #                     window_size: Number of samples per window
    #                     step: Step between windows
    #                     value_columns: Optional columns used by WindowBuilder
    #                     include_meta: Keep metadata columns in flattened windows
    #                     drop_non_numeric: Drop non-numeric columns in output matrix
    #                     fill_method: Missing-value fill method (None/zero/ffill/median)
    # @param[out]         out: Feature DataFrame
    #
    # @callsequence       @startuml
    #                     title build_feature_matrix
    #                     start
    #                     :validate_dataframe(df);
    #                     :validate window_size and step;
    #                     :WindowBuilder.build_window_sequence(...);
    #                     :WindowBuilder.windows_to_feature_frame(...);
    #                     if (drop_non_numeric?) then (yes)
    #                       :drop_non_numeric_features(X);
    #                     endif
    #                     :fill_feature_matrix(X, fill_method);
    #                     :return X;
    #                     end
    #                     @enduml
    # ***********************************************************************************************************************
    @staticmethod
    def build_feature_matrix(
        df: pd.DataFrame,
        window_size: int,
        step: int,
        value_columns: list[str] | None = None,
        include_meta: bool = False,
        drop_non_numeric: bool = True,
        fill_method: str | None = None,
    ) -> pd.DataFrame:
        validate_dataframe(df)
        if window_size <= 0:
            raise ValueError("window_size must be > 0")
        if step <= 0:
            raise ValueError("step must be > 0")

        windows = WindowBuilder.build_window_sequence(
            df=df,
            window_size=window_size,
            step=step,
            value_columns=value_columns,
        )

        X = WindowBuilder.windows_to_feature_frame(
            windows,
            include_meta=include_meta,
        )

        if drop_non_numeric and not include_meta:
            X = FeaturePipeline.drop_non_numeric_features(X)
        elif drop_non_numeric:
            # Keep metadata if requested, but still reduce to numeric otherwise.
            numeric = FeaturePipeline.drop_non_numeric_features(X)
            meta_cols = [c for c in X.columns if c.startswith("window_meta__")]
            if meta_cols:
                X = pd.concat([X[meta_cols], numeric], axis=1)
            else:
                X = numeric

        X = FeaturePipeline.fill_feature_matrix(X, method=fill_method)
        return X

    # ***********************************************************************************************************************
    # Functionname:       FeaturePipeline.validate_feature_matrix(X, require_numeric, allow_nan)
    #
    # @brief              Validate feature matrix constraints for downstream ML/TDA.
    # @pre                X is expected to be a DataFrame.
    # @post               Raises ValueError on validation failures.
    # @param[in]          X: Feature DataFrame to validate
    #                     require_numeric: Enforce all columns numeric
    #                     allow_nan: Allow NaN values if True
    # @param[out]         None
    #
    # @callsequence       @startuml
    #                     title validate_feature_matrix
    #                     start
    #                     :validate_dataframe(X);
    #                     if (X empty?) then (yes)
    #                       :raise ValueError;
    #                     endif
    #                     if (require_numeric?) then (yes)
    #                       :detect non-numeric columns;
    #                       if (any?) then (yes)
    #                         :raise ValueError;
    #                       endif
    #                     endif
    #                     if (allow_nan == False?) then (yes)
    #                       :check NaN presence;
    #                       if (NaN exists?) then (yes)
    #                         :raise ValueError;
    #                       endif
    #                     endif
    #                     end
    #                     @enduml
    # ***********************************************************************************************************************
    @staticmethod
    def validate_feature_matrix(
        X: pd.DataFrame,
        require_numeric: bool = True,
        allow_nan: bool = True,
    ) -> None:
        validate_dataframe(X)

        if X.shape[1] == 0:
            raise ValueError("Feature matrix has zero columns.")

        if require_numeric:
            non_numeric_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
            if non_numeric_cols:
                raise ValueError(
                    f"Feature matrix has non-numeric columns: {sorted(non_numeric_cols)}"
                )

        if not allow_nan and X.isna().any().any():
            raise ValueError("Feature matrix contains NaN values.")

    # ***********************************************************************************************************************
    # Functionname:       FeaturePipeline.drop_non_numeric_features(X)
    #
    # @brief              Return copy with only numeric feature columns.
    # @pre                X is a DataFrame.
    # @post               Output includes only numeric dtypes.
    # @param[in]          X: Input feature DataFrame
    # @param[out]         out: Numeric-only feature DataFrame
    # ***********************************************************************************************************************
    @staticmethod
    def drop_non_numeric_features(X: pd.DataFrame) -> pd.DataFrame:
        validate_dataframe(X)
        return X.select_dtypes(include=[np.number]).copy()

    # ***********************************************************************************************************************
    # Functionname:       FeaturePipeline.fill_feature_matrix(X, method)
    #
    # @brief              Fill missing values in feature matrix using supported strategy.
    # @pre                X is a DataFrame; method in {None, zero, ffill, median}.
    # @post               Returns filled DataFrame copy.
    # @param[in]          X: Input feature DataFrame
    #                     method: Fill strategy
    # @param[out]         out: Filled DataFrame
    # ***********************************************************************************************************************
    @staticmethod
    def fill_feature_matrix(
        X: pd.DataFrame,
        method: str | None = None,
    ) -> pd.DataFrame:
        validate_dataframe(X)

        out = X.copy()

        if method is None:
            return out
        if method == "zero":
            return out.fillna(0)
        if method == "ffill":
            return out.ffill().bfill()
        if method == "median":
            numeric_cols = out.select_dtypes(include=[np.number]).columns
            medians = out[numeric_cols].median()
            out[numeric_cols] = out[numeric_cols].fillna(medians)
            return out

        raise ValueError("Unsupported fill method. Use one of: None, 'zero', 'ffill', 'median'.")

    # ***********************************************************************************************************************
    # Functionname:       FeaturePipeline.build_pipeline_report(X)
    #
    # @brief              Build diagnostics summary for feature matrix quality.
    # @pre                X is a DataFrame.
    # @post               Returns dict with matrix shape and quality indicators.
    # @param[in]          X: Feature DataFrame
    # @param[out]         out: Diagnostics dict
    # ***********************************************************************************************************************
    @staticmethod
    def build_pipeline_report(X: pd.DataFrame) -> dict:
        validate_dataframe(X)

        non_numeric_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
        if X.shape[0] == 0 or X.shape[1] == 0:
            missing_ratio = 0.0
        else:
            missing_ratio = float(X.isna().sum().sum()) / float(X.shape[0] * X.shape[1])

        constant_columns = [
            col for col in X.columns
            if X[col].nunique(dropna=False) <= 1
        ]

        return {
            "n_rows": int(X.shape[0]),
            "n_features": int(X.shape[1]),
            "missing_ratio": float(missing_ratio),
            "non_numeric_columns": non_numeric_cols,
            "constant_columns": constant_columns,
        }


# ---------------------------------------------------------------------------
# Module-level convenience wrappers
# ---------------------------------------------------------------------------


def build_feature_matrix(
    df: pd.DataFrame,
    window_size: int,
    step: int,
    value_columns: list[str] | None = None,
    include_meta: bool = False,
    drop_non_numeric: bool = True,
    fill_method: str | None = None,
) -> pd.DataFrame:
    return FeaturePipeline.build_feature_matrix(
        df=df,
        window_size=window_size,
        step=step,
        value_columns=value_columns,
        include_meta=include_meta,
        drop_non_numeric=drop_non_numeric,
        fill_method=fill_method,
    )


def validate_feature_matrix(
    X: pd.DataFrame,
    require_numeric: bool = True,
    allow_nan: bool = True,
) -> None:
    FeaturePipeline.validate_feature_matrix(
        X,
        require_numeric=require_numeric,
        allow_nan=allow_nan,
    )


def drop_non_numeric_features(X: pd.DataFrame) -> pd.DataFrame:
    return FeaturePipeline.drop_non_numeric_features(X)


def fill_feature_matrix(
    X: pd.DataFrame,
    method: str | None = None,
) -> pd.DataFrame:
    return FeaturePipeline.fill_feature_matrix(X, method=method)


def build_pipeline_report(X: pd.DataFrame) -> dict:
    return FeaturePipeline.build_pipeline_report(X)


__all__ = [
    "FeaturePipeline",
    "build_feature_matrix",
    "validate_feature_matrix",
    "drop_non_numeric_features",
    "fill_feature_matrix",
    "build_pipeline_report",
]
