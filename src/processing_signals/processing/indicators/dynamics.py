# \file **********************************************************************
# COMPANY:            ELATIN
# PROJECT:            SIGNAL_ANALYSIS
# COMPONENT:          INDICATORS
# MODULE NAME:        dynamics.py
# DESCRIPTION:        @brief Higher-order temporal dynamics features.
#                     Computes velocity, acceleration, jerk, and curvature
#                     for any generic scalar temporal signal.
# CREATION DATE:      06.05.2026
# VERSION:            $Revision: 0.1$
# CHANGES:            06.05.2026 - Initial implementation.
# *****************************************************************************

from __future__ import annotations

import numpy as np
import pandas as pd

from signal_analysis.core.schema import validate_dataframe


class DynamicsIndicators:
    """
    Higher-order temporal dynamics feature builders.

    Computes velocity (1st derivative), acceleration (2nd derivative),
    jerk (3rd derivative), and local curvature for generic scalar signals.

    Domain-agnostic: works for price, position, velocity, temperature,
    pressure, flow rate, vibration amplitude, or any scalar time series.
    """

    # ***********************************************************************************************************************
    # Functionname:       DynamicsIndicators.compute_velocity(series, periods)
    #
    # @brief              Compute first discrete difference (velocity proxy).
    # @pre                series is array-like; periods > 0.
    # @post               Returns Series of same length; first 'periods' values are NaN.
    # @param[in]          series: Input 1-D signal
    #                     periods: Lag for finite difference
    # @param[out]         out: Velocity Series
    # ***********************************************************************************************************************
    @staticmethod
    def compute_velocity(
        series: pd.Series,
        periods: int = 1,
    ) -> pd.Series:
        """Compute first finite difference: x[t] - x[t-periods]."""
        s = pd.Series(series, dtype=float)
        result = s.diff(periods)
        result.index = s.index
        return result

    # ***********************************************************************************************************************
    # Functionname:       DynamicsIndicators.compute_acceleration(series, periods)
    #
    # @brief              Compute second discrete difference (acceleration proxy).
    # @pre                series is array-like; periods > 0.
    # @post               Returns Series with first 2*periods values as NaN.
    # @param[in]          series: Input 1-D signal
    #                     periods: Lag for each finite difference step
    # @param[out]         out: Acceleration Series
    # ***********************************************************************************************************************
    @staticmethod
    def compute_acceleration(
        series: pd.Series,
        periods: int = 1,
    ) -> pd.Series:
        """Compute second finite difference (diff of diff)."""
        s = pd.Series(series, dtype=float)
        result = s.diff(periods).diff(periods)
        result.index = s.index
        return result

    # ***********************************************************************************************************************
    # Functionname:       DynamicsIndicators.compute_jerk(series, periods)
    #
    # @brief              Compute third discrete difference (jerk proxy).
    # @pre                series is array-like; periods > 0.
    # @post               Returns Series with first 3*periods values as NaN.
    # @param[in]          series: Input 1-D signal
    #                     periods: Lag for each finite difference step
    # @param[out]         out: Jerk Series
    # ***********************************************************************************************************************
    @staticmethod
    def compute_jerk(
        series: pd.Series,
        periods: int = 1,
    ) -> pd.Series:
        """Compute third finite difference (diff of diff of diff)."""
        s = pd.Series(series, dtype=float)
        result = s.diff(periods).diff(periods).diff(periods)
        result.index = s.index
        return result

    # ***********************************************************************************************************************
    # Functionname:       DynamicsIndicators.compute_curvature(series)
    #
    # @brief              Compute discrete local curvature proxy.
    # @pre                series is array-like with at least 3 elements.
    # @post               Returns Series with first 2 values as NaN.
    # @param[in]          series: Input 1-D signal
    # @param[out]         out: Curvature Series
    #
    # @note               Curvature proxy: d2 / (1 + d1^2)^(3/2)
    #                     where d1 = first difference, d2 = second difference.
    # ***********************************************************************************************************************
    @staticmethod
    def compute_curvature(
        series: pd.Series,
    ) -> pd.Series:
        """
        Compute discrete local curvature proxy.

        Formula: d2 / (1 + d1^2)^(3/2)
        where d1 is the first finite difference and d2 is the second.
        """
        s = pd.Series(series, dtype=float)
        d1 = s.diff()
        d2 = d1.diff()
        result = d2 / (1.0 + d1 ** 2).pow(1.5)
        result.index = s.index
        return result

    # ***********************************************************************************************************************
    # Functionname:       DynamicsIndicators.compute_dynamic_profile(series, periods)
    #
    # @brief              Compute all four dynamics features in one call.
    # @pre                series is array-like; periods > 0.
    # @post               Returns DataFrame with velocity, acceleration, jerk, curvature.
    # @param[in]          series: Input 1-D signal
    #                     periods: Lag for finite difference
    # @param[out]         out: Dynamics DataFrame
    # ***********************************************************************************************************************
    @staticmethod
    def compute_dynamic_profile(
        series: pd.Series,
        periods: int = 1,
    ) -> pd.DataFrame:
        """
        Compute all four dynamics features for a single series.

        Returns DataFrame with columns:
          velocity, acceleration, jerk, curvature
        """
        s = pd.Series(series, dtype=float)
        return pd.DataFrame(
            {
                "velocity": DynamicsIndicators.compute_velocity(s, periods=periods),
                "acceleration": DynamicsIndicators.compute_acceleration(s, periods=periods),
                "jerk": DynamicsIndicators.compute_jerk(s, periods=periods),
                "curvature": DynamicsIndicators.compute_curvature(s),
            },
            index=s.index,
        )

    # ***********************************************************************************************************************
    # Functionname:       DynamicsIndicators.add_dynamics_features(df, columns, periods)
    #
    # @brief              Add dynamics feature columns for all specified columns.
    # @pre                df is a DataFrame; periods > 0.
    # @post               Returns copy with new columns; input is not mutated.
    # @param[in]          df: Input DataFrame
    #                     columns: List of column names to process
    #                     periods: Lag for finite difference
    #                     prefix: Optional column name prefix
    # @param[out]         out: Enriched DataFrame copy
    # ***********************************************************************************************************************
    @staticmethod
    def add_dynamics_features(
        df: pd.DataFrame,
        columns: list[str],
        periods: int = 1,
        prefix: str = "",
    ) -> pd.DataFrame:
        """
        Add dynamics feature columns for specified numeric columns.

        Silently ignores missing or non-numeric columns.
        Does not mutate input DataFrame.

        For each eligible column, adds:
          {prefix}{col}_velocity
          {prefix}{col}_acceleration
          {prefix}{col}_jerk
          {prefix}{col}_curvature
        """
        validate_dataframe(df)
        out = df.copy()

        for col in columns:
            if col not in df.columns:
                continue
            if not pd.api.types.is_numeric_dtype(df[col]):
                continue

            s = df[col]
            p = prefix

            out[f"{p}{col}_velocity"] = DynamicsIndicators.compute_velocity(
                s, periods=periods
            )
            out[f"{p}{col}_acceleration"] = DynamicsIndicators.compute_acceleration(
                s, periods=periods
            )
            out[f"{p}{col}_jerk"] = DynamicsIndicators.compute_jerk(
                s, periods=periods
            )
            out[f"{p}{col}_curvature"] = DynamicsIndicators.compute_curvature(s)

        return out


# ---------------------------------------------------------------------------
# Module-level convenience wrappers
# ---------------------------------------------------------------------------


def compute_velocity(series: pd.Series, periods: int = 1) -> pd.Series:
    """Wrapper for DynamicsIndicators.compute_velocity."""
    return DynamicsIndicators.compute_velocity(series, periods=periods)


def compute_acceleration(series: pd.Series, periods: int = 1) -> pd.Series:
    """Wrapper for DynamicsIndicators.compute_acceleration."""
    return DynamicsIndicators.compute_acceleration(series, periods=periods)


def compute_jerk(series: pd.Series, periods: int = 1) -> pd.Series:
    """Wrapper for DynamicsIndicators.compute_jerk."""
    return DynamicsIndicators.compute_jerk(series, periods=periods)


def compute_curvature(series: pd.Series) -> pd.Series:
    """Wrapper for DynamicsIndicators.compute_curvature."""
    return DynamicsIndicators.compute_curvature(series)


def compute_dynamic_profile(series: pd.Series, periods: int = 1) -> pd.DataFrame:
    """Wrapper for DynamicsIndicators.compute_dynamic_profile."""
    return DynamicsIndicators.compute_dynamic_profile(series, periods=periods)


def add_dynamics_features(
    df: pd.DataFrame,
    columns: list[str],
    periods: int = 1,
    prefix: str = "",
) -> pd.DataFrame:
    """Wrapper for DynamicsIndicators.add_dynamics_features."""
    return DynamicsIndicators.add_dynamics_features(
        df, columns=columns, periods=periods, prefix=prefix
    )


__all__ = [
    "DynamicsIndicators",
    "compute_velocity",
    "compute_acceleration",
    "compute_jerk",
    "compute_curvature",
    "compute_dynamic_profile",
    "add_dynamics_features",
]
