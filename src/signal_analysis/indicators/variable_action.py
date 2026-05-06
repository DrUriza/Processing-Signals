# \file **********************************************************************
# COMPANY:            ELATIN
# PROJECT:            SIGNAL_ANALYSIS
# COMPONENT:          INDICATORS
# MODULE NAME:        variable_action.py
# DESCRIPTION:        @brief Generic OHLC geometry and state features (domain-agnostic)
# CREATION DATE:      06.05.2026
# VERSION:            $Revision: 0.1$
# CHANGES:            06.05.2026 - Initial variable-action OHLC geometry indicators.
# *****************************************************************************

from __future__ import annotations

import numpy as np
import pandas as pd

from signal_analysis.core.schema import validate_ohlc_columns


class VariableActionIndicators:
    """
    Domain-agnostic OHLC geometry and state classification indicators.

    OHLC structure represents any temporal variable with high/low bounds:
    price, pressure, temperature, flow, position, energy, vibration, etc.

    No trading, order-book, or domain-specific terminology.
    """

    # ***********************************************************************************************************************
    # Functionname:       VariableActionIndicators.compute_ohlc_geometry(df, eps)
    #
    # @brief              Compute geometric features from OHLC data.
    # @pre                df contains valid OHLC columns; eps > 0.
    # @post               Returns DataFrame with geometry columns aligned to input index.
    # @param[in]          df: OHLC DataFrame
    #                     eps: Numerical stability factor
    # @param[out]         out: Geometry features DataFrame
    #
    # @callsequence       @startuml
    #                     title VariableActionIndicators.compute_ohlc_geometry
    #                     start
    #                     :Call validate_ohlc_columns(df);
    #                     :Compute range = high - low;
    #                     :Compute body_size = |close - open|;
    #                     :Compute body_direction = sign(close - open);
    #                     :Compute upper_wick, lower_wick;
    #                     :Compute ratios (normalized by range + eps);
    #                     :Compute wick_imbalance;
    #                     :Compute close_position;
    #                     :Compute gap, return, log_return;
    #                     :Return DataFrame with all geometry columns;
    #                     end
    #                     @enduml
    # ***********************************************************************************************************************
    @staticmethod
    def compute_ohlc_geometry(df: pd.DataFrame, eps: float = 1e-9) -> pd.DataFrame:
        """
        Compute OHLC geometric features.

        Includes range, body dimensions, wicks, positions, returns, gaps.

        Args:
            df: DataFrame with OHLC columns.
            eps: Numerical stability factor. Defaults to 1e-9.

        Returns:
            DataFrame with geometry features aligned to input index.

        Raises:
            ValueError: If OHLC columns are missing or non-numeric.
        """
        validate_ohlc_columns(df)

        result = {}

        # Range and body
        result["range"] = df["high"] - df["low"]
        result["body_size"] = (df["close"] - df["open"]).abs()
        result["body_direction"] = np.sign(df["close"] - df["open"])

        # Wicks
        result["upper_wick"] = df["high"] - df[["open", "close"]].max(axis=1)
        result["lower_wick"] = df[["open", "close"]].min(axis=1) - df["low"]

        # Ratios (normalized)
        range_plus_eps = result["range"] + eps
        result["body_ratio"] = result["body_size"] / range_plus_eps
        result["upper_wick_ratio"] = result["upper_wick"] / range_plus_eps
        result["lower_wick_ratio"] = result["lower_wick"] / range_plus_eps
        result["wick_imbalance"] = result["upper_wick"] - result["lower_wick"]
        result["wick_imbalance_ratio"] = result["wick_imbalance"] / range_plus_eps

        # Close position within range
        result["close_position"] = (df["close"] - df["low"]) / range_plus_eps

        # Gaps and returns
        result["gap"] = df["open"] - df["close"].shift(1)
        result["return"] = df["close"].pct_change()
        result["log_return"] = np.log(df["close"] / df["close"].shift(1))

        return pd.DataFrame(result, index=df.index)

    # ***********************************************************************************************************************
    # Functionname:       VariableActionIndicators.classify_ohlc_geometry(df, doji_body_ratio,
    #                              strong_body_ratio, long_wick_ratio)
    #
    # @brief              Classify OHLC into categorical and boolean states.
    # @pre                df contains OHLC columns; thresholds are in [0, 1].
    # @post               Returns DataFrame with body_state, direction_state, rejection_state and flags.
    # @param[in]          df: OHLC DataFrame
    #                     doji_body_ratio: Body/range threshold for doji (default 0.1)
    #                     strong_body_ratio: Body/range threshold for strong (default 0.6)
    #                     long_wick_ratio: Wick/range threshold for rejection (default 0.45)
    # @param[out]         out: Classified states DataFrame
    #
    # @callsequence       @startuml
    #                     title VariableActionIndicators.classify_ohlc_geometry
    #                     start
    #                     :Call compute_ohlc_geometry(df);
    #                     :Classify body_state based on body_ratio thresholds;
    #                     :Classify direction_state from body_direction;
    #                     :Classify rejection_state from wick comparisons;
    #                     :Create boolean flags for doji, strong_body, rejections;
    #                     :Return DataFrame with all classifications;
    #                     end
    #                     @enduml
    # ***********************************************************************************************************************
    @staticmethod
    def classify_ohlc_geometry(
        df: pd.DataFrame,
        doji_body_ratio: float = 0.1,
        strong_body_ratio: float = 0.6,
        long_wick_ratio: float = 0.45,
    ) -> pd.DataFrame:
        """
        Classify OHLC into categorical and boolean states.

        States:
        - body_state: "doji", "weak_body", "normal_body", "strong_body"
        - direction_state: "up", "down", "neutral"
        - rejection_state: "upper_rejection", "lower_rejection", "both_rejection", "none"

        Args:
            df: DataFrame with OHLC columns.
            doji_body_ratio: Body/range ratio threshold for doji. Defaults to 0.1.
            strong_body_ratio: Body/range ratio threshold for strong. Defaults to 0.6.
            long_wick_ratio: Wick/range ratio threshold for rejection. Defaults to 0.45.

        Returns:
            DataFrame with classification columns.

        Raises:
            ValueError: If OHLC columns are missing or non-numeric.
        """
        geom = VariableActionIndicators.compute_ohlc_geometry(df)

        result = {}

        # Body state
        body_ratio = geom["body_ratio"]
        body_state = pd.Series("normal_body", index=df.index, dtype="object")
        body_state[body_ratio <= doji_body_ratio] = "doji"
        body_state[
            (body_ratio > doji_body_ratio) & (body_ratio < strong_body_ratio)
        ] = "weak_body"
        body_state[body_ratio >= strong_body_ratio] = "strong_body"
        result["body_state"] = body_state

        # Direction state
        direction_state = pd.Series("neutral", index=df.index, dtype="object")
        direction_state[geom["body_direction"] > 0] = "up"
        direction_state[geom["body_direction"] < 0] = "down"
        result["direction_state"] = direction_state

        # Rejection state
        upper_wick_ratio = geom["upper_wick_ratio"]
        lower_wick_ratio = geom["lower_wick_ratio"]

        is_upper_rej = upper_wick_ratio >= long_wick_ratio
        is_lower_rej = lower_wick_ratio >= long_wick_ratio

        rejection_state = pd.Series("none", index=df.index, dtype="object")
        rejection_state[is_upper_rej & ~is_lower_rej] = "upper_rejection"
        rejection_state[is_lower_rej & ~is_upper_rej] = "lower_rejection"
        rejection_state[is_upper_rej & is_lower_rej] = "both_rejection"
        result["rejection_state"] = rejection_state

        # Boolean flags
        result["is_doji"] = body_ratio <= doji_body_ratio
        result["is_strong_body"] = body_ratio >= strong_body_ratio
        result["has_upper_rejection"] = is_upper_rej
        result["has_lower_rejection"] = is_lower_rej

        return pd.DataFrame(result, index=df.index)

    # ***********************************************************************************************************************
    # Functionname:       VariableActionIndicators.add_ohlc_geometry_features(df, eps)
    #
    # @brief              Add OHLC geometry features to DataFrame copy.
    # @pre                df contains OHLC columns.
    # @post               Returns copy with geometry and classification columns concatenated.
    # @param[in]          df: OHLC DataFrame
    #                     eps: Numerical stability factor
    # @param[out]         out: DataFrame with original and geometry columns
    #
    # @callsequence       @startuml
    #                     title VariableActionIndicators.add_ohlc_geometry_features
    #                     start
    #                     :Make copy of input df;
    #                     :Compute geometry features;
    #                     :Compute classification features;
    #                     :Concatenate all columns;
    #                     :Return enriched copy (no mutation);
    #                     end
    #                     @enduml
    # ***********************************************************************************************************************
    @staticmethod
    def add_ohlc_geometry_features(df: pd.DataFrame, eps: float = 1e-9) -> pd.DataFrame:
        """
        Add OHLC geometry and classification features to DataFrame.

        Returns a copy; does not mutate input.

        Args:
            df: DataFrame with OHLC columns.
            eps: Numerical stability factor. Defaults to 1e-9.

        Returns:
            Copy of df with geometry and classification columns added.

        Raises:
            ValueError: If OHLC columns are missing or non-numeric.
        """
        validate_ohlc_columns(df)

        out = df.copy()
        geom = VariableActionIndicators.compute_ohlc_geometry(df, eps=eps)
        classified = VariableActionIndicators.classify_ohlc_geometry(df)

        out = pd.concat([out, geom, classified], axis=1)

        return out


# ---------------------------------------------------------------------------
# Module-level convenience wrappers
# ---------------------------------------------------------------------------


def compute_ohlc_geometry(df: pd.DataFrame, eps: float = 1e-9) -> pd.DataFrame:
    """Wrapper for VariableActionIndicators.compute_ohlc_geometry."""
    return VariableActionIndicators.compute_ohlc_geometry(df, eps=eps)


def classify_ohlc_geometry(
    df: pd.DataFrame,
    doji_body_ratio: float = 0.1,
    strong_body_ratio: float = 0.6,
    long_wick_ratio: float = 0.45,
) -> pd.DataFrame:
    """Wrapper for VariableActionIndicators.classify_ohlc_geometry."""
    return VariableActionIndicators.classify_ohlc_geometry(
        df,
        doji_body_ratio=doji_body_ratio,
        strong_body_ratio=strong_body_ratio,
        long_wick_ratio=long_wick_ratio,
    )


def add_ohlc_geometry_features(df: pd.DataFrame, eps: float = 1e-9) -> pd.DataFrame:
    """Wrapper for VariableActionIndicators.add_ohlc_geometry_features."""
    return VariableActionIndicators.add_ohlc_geometry_features(df, eps=eps)


__all__ = [
    "VariableActionIndicators",
    "compute_ohlc_geometry",
    "classify_ohlc_geometry",
    "add_ohlc_geometry_features",
]
