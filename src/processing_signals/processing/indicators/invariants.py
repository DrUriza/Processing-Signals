# \file **********************************************************************
# COMPANY:            ELATIN
# PROJECT:            SIGNAL_ANALYSIS
# COMPONENT:          INDICATORS
# MODULE NAME:        invariants.py
# DESCRIPTION:        @brief Scale-invariant and robust normalized features
#                     for generic temporal signals (domain-agnostic).
# CREATION DATE:      06.05.2026
# VERSION:            $Revision: 0.1$
# CHANGES:            06.05.2026 - Initial implementation.
# *****************************************************************************

from __future__ import annotations

import numpy as np
import pandas as pd

from signal_analysis.core.schema import validate_dataframe


class InvariantIndicators:
    """
    Scale-invariant and robust normalized feature builders.

    All methods operate on generic 1-D temporal signals.
    No domain-specific terminology. Suitable for trading, robotics,
    fluids, radar, or mathematical time series.
    """

    # ***********************************************************************************************************************
    # Functionname:       InvariantIndicators.compute_rolling_zscore(series, window, eps)
    #
    # @brief              Compute rolling standard z-score.
    # @pre                series is array-like; window > 0.
    # @post               Returns Series of same length; NaN where window not met.
    # @param[in]          series: Input 1-D signal
    #                     window: Rolling window size
    #                     eps: Numerical stability factor
    # @param[out]         out: Rolling z-score Series
    # ***********************************************************************************************************************
    @staticmethod
    def compute_rolling_zscore(
        series: pd.Series,
        window: int = 20,
        eps: float = 1e-9,
    ) -> pd.Series:
        """Compute rolling mean-std z-score: (x - mean) / (std + eps)."""
        s = pd.Series(series, dtype=float)
        mean = s.rolling(window=window).mean()
        std = s.rolling(window=window).std()
        result = (s - mean) / (std + eps)
        result.index = s.index
        return result

    # ***********************************************************************************************************************
    # Functionname:       InvariantIndicators.compute_robust_zscore(series, window, eps)
    #
    # @brief              Compute robust z-score using rolling median and MAD.
    # @pre                series is array-like; window > 0.
    # @post               Returns Series of same length and index.
    # @param[in]          series: Input 1-D signal
    #                     window: Rolling window size
    #                     eps: Numerical stability factor
    # @param[out]         out: Robust z-score Series
    # ***********************************************************************************************************************
    @staticmethod
    def compute_robust_zscore(
        series: pd.Series,
        window: int = 20,
        eps: float = 1e-9,
    ) -> pd.Series:
        """
        Compute robust z-score using rolling median and MAD.

        z = (x - median) / (1.4826 * MAD + eps)

        The 1.4826 factor makes MAD consistent with std for Gaussian signals.
        """
        s = pd.Series(series, dtype=float)
        median = s.rolling(window=window).median()
        mad = (s - median).abs().rolling(window=window).median()
        result = (s - median) / (1.4826 * mad + eps)
        result.index = s.index
        return result

    # ***********************************************************************************************************************
    # Functionname:       InvariantIndicators.compute_local_ratio(series, window, eps)
    #
    # @brief              Compute ratio of signal to rolling mean of absolute values.
    # @pre                series is array-like; window > 0.
    # @post               Returns Series with same index; handles positive and negative.
    # @param[in]          series: Input 1-D signal
    #                     window: Rolling window size
    #                     eps: Numerical stability factor
    # @param[out]         out: Local ratio Series
    # ***********************************************************************************************************************
    @staticmethod
    def compute_local_ratio(
        series: pd.Series,
        window: int = 20,
        eps: float = 1e-9,
    ) -> pd.Series:
        """Compute x / rolling_mean(|x|) — scale-invariant local ratio."""
        s = pd.Series(series, dtype=float)
        rolling_mean_abs = s.abs().rolling(window=window).mean()
        result = s / (rolling_mean_abs + eps)
        result.index = s.index
        return result

    # ***********************************************************************************************************************
    # Functionname:       InvariantIndicators.compute_log_ratio(series, reference, window, eps)
    #
    # @brief              Compute log ratio of signal to reference baseline.
    # @pre                series is array-like; window > 0.
    # @post               Returns Series with same index; zeros handled via eps.
    # @param[in]          series: Input 1-D signal
    #                     reference: Optional reference Series; uses rolling mean |x| if None
    #                     window: Rolling window for auto-reference
    #                     eps: Numerical stability factor
    # @param[out]         out: Log ratio Series
    # ***********************************************************************************************************************
    @staticmethod
    def compute_log_ratio(
        series: pd.Series,
        reference: pd.Series | None = None,
        window: int = 20,
        eps: float = 1e-9,
    ) -> pd.Series:
        """
        Compute log((|x| + eps) / (|ref| + eps)).

        If reference is None, rolling mean of |x| is used as reference.
        """
        s = pd.Series(series, dtype=float)
        if reference is None:
            ref = s.abs().rolling(window=window).mean()
        else:
            ref = pd.Series(reference, dtype=float)
        result = np.log((s.abs() + eps) / (ref.abs() + eps))
        result.index = s.index
        return result

    # ***********************************************************************************************************************
    # Functionname:       InvariantIndicators.compute_relative_change(series, periods, eps)
    #
    # @brief              Compute relative change normalized by absolute prior value.
    # @pre                series is array-like; periods > 0.
    # @post               Returns Series of same length and index.
    # @param[in]          series: Input 1-D signal
    #                     periods: Lag for difference
    #                     eps: Numerical stability factor
    # @param[out]         out: Relative change Series
    # ***********************************************************************************************************************
    @staticmethod
    def compute_relative_change(
        series: pd.Series,
        periods: int = 1,
        eps: float = 1e-9,
    ) -> pd.Series:
        """Compute (x - x[t-n]) / (|x[t-n]| + eps) — relative change."""
        s = pd.Series(series, dtype=float)
        shifted = s.shift(periods)
        result = (s - shifted) / (shifted.abs() + eps)
        result.index = s.index
        return result

    # ***********************************************************************************************************************
    # Functionname:       InvariantIndicators.compute_rolling_energy(series, window)
    #
    # @brief              Compute rolling mean of squared signal (instantaneous energy proxy).
    # @pre                series is array-like; window > 0.
    # @post               Returns non-negative Series with same index.
    # @param[in]          series: Input 1-D signal
    #                     window: Rolling window size
    # @param[out]         out: Rolling energy Series
    # ***********************************************************************************************************************
    @staticmethod
    def compute_rolling_energy(
        series: pd.Series,
        window: int = 20,
    ) -> pd.Series:
        """Compute rolling mean of x^2 (energy proxy)."""
        s = pd.Series(series, dtype=float)
        result = (s ** 2).rolling(window=window).mean()
        result.index = s.index
        return result

    # ***********************************************************************************************************************
    # Functionname:       InvariantIndicators.compute_normalized_energy(series, window, eps)
    #
    # @brief              Normalize rolling energy by its own rolling median.
    # @pre                series is array-like; window > 0.
    # @post               Returns finite Series; avoids near-zero median via eps.
    # @param[in]          series: Input 1-D signal
    #                     window: Rolling window size
    #                     eps: Numerical stability factor
    # @param[out]         out: Normalized energy Series
    # ***********************************************************************************************************************
    @staticmethod
    def compute_normalized_energy(
        series: pd.Series,
        window: int = 20,
        eps: float = 1e-9,
    ) -> pd.Series:
        """Compute rolling_energy / rolling_median(rolling_energy)."""
        s = pd.Series(series, dtype=float)
        energy = (s ** 2).rolling(window=window).mean()
        median_energy = energy.rolling(window=window).median()
        result = energy / (median_energy + eps)
        result.index = s.index
        return result

    # ***********************************************************************************************************************
    # Functionname:       InvariantIndicators.compute_minmax_position(series, window, eps)
    #
    # @brief              Compute position of x within local rolling [min, max] range.
    # @pre                series is array-like; window > 0.
    # @post               Returns Series in [0, 1] for non-constant windows.
    # @param[in]          series: Input 1-D signal
    #                     window: Rolling window size
    #                     eps: Numerical stability factor
    # @param[out]         out: Minmax position Series
    # ***********************************************************************************************************************
    @staticmethod
    def compute_minmax_position(
        series: pd.Series,
        window: int = 20,
        eps: float = 1e-9,
    ) -> pd.Series:
        """Compute (x - rolling_min) / (rolling_max - rolling_min + eps)."""
        s = pd.Series(series, dtype=float)
        rolling_min = s.rolling(window=window).min()
        rolling_max = s.rolling(window=window).max()
        result = (s - rolling_min) / (rolling_max - rolling_min + eps)
        result.index = s.index
        return result

    # ***********************************************************************************************************************
    # Functionname:       InvariantIndicators.add_invariant_features(df, columns, window, prefix, eps)
    #
    # @brief              Add all invariant feature columns for specified columns.
    # @pre                df is a DataFrame; window > 0.
    # @post               Returns copy of df with new columns; input is not mutated.
    # @param[in]          df: Input DataFrame
    #                     columns: List of column names to process
    #                     window: Rolling window size
    #                     prefix: Optional column name prefix
    #                     eps: Numerical stability factor
    # @param[out]         out: Enriched DataFrame copy
    # ***********************************************************************************************************************
    @staticmethod
    def add_invariant_features(
        df: pd.DataFrame,
        columns: list[str],
        window: int = 20,
        prefix: str = "",
        eps: float = 1e-9,
    ) -> pd.DataFrame:
        """
        Add invariant feature columns for all specified numeric columns.

        Silently ignores missing or non-numeric columns.
        Does not mutate input DataFrame.

        For each eligible column, adds:
          {prefix}{col}_zscore
          {prefix}{col}_robust_zscore
          {prefix}{col}_local_ratio
          {prefix}{col}_log_ratio
          {prefix}{col}_relative_change
          {prefix}{col}_rolling_energy
          {prefix}{col}_normalized_energy
          {prefix}{col}_minmax_position
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

            out[f"{p}{col}_zscore"] = InvariantIndicators.compute_rolling_zscore(
                s, window=window, eps=eps
            )
            out[f"{p}{col}_robust_zscore"] = InvariantIndicators.compute_robust_zscore(
                s, window=window, eps=eps
            )
            out[f"{p}{col}_local_ratio"] = InvariantIndicators.compute_local_ratio(
                s, window=window, eps=eps
            )
            out[f"{p}{col}_log_ratio"] = InvariantIndicators.compute_log_ratio(
                s, window=window, eps=eps
            )
            out[f"{p}{col}_relative_change"] = InvariantIndicators.compute_relative_change(
                s, eps=eps
            )
            out[f"{p}{col}_rolling_energy"] = InvariantIndicators.compute_rolling_energy(
                s, window=window
            )
            out[f"{p}{col}_normalized_energy"] = InvariantIndicators.compute_normalized_energy(
                s, window=window, eps=eps
            )
            out[f"{p}{col}_minmax_position"] = InvariantIndicators.compute_minmax_position(
                s, window=window, eps=eps
            )

        return out


# ---------------------------------------------------------------------------
# Module-level convenience wrappers
# ---------------------------------------------------------------------------


def compute_rolling_zscore(
    series: pd.Series,
    window: int = 20,
    eps: float = 1e-9,
) -> pd.Series:
    """Wrapper for InvariantIndicators.compute_rolling_zscore."""
    return InvariantIndicators.compute_rolling_zscore(series, window=window, eps=eps)


def compute_robust_zscore(
    series: pd.Series,
    window: int = 20,
    eps: float = 1e-9,
) -> pd.Series:
    """Wrapper for InvariantIndicators.compute_robust_zscore."""
    return InvariantIndicators.compute_robust_zscore(series, window=window, eps=eps)


def compute_local_ratio(
    series: pd.Series,
    window: int = 20,
    eps: float = 1e-9,
) -> pd.Series:
    """Wrapper for InvariantIndicators.compute_local_ratio."""
    return InvariantIndicators.compute_local_ratio(series, window=window, eps=eps)


def compute_log_ratio(
    series: pd.Series,
    reference: pd.Series | None = None,
    window: int = 20,
    eps: float = 1e-9,
) -> pd.Series:
    """Wrapper for InvariantIndicators.compute_log_ratio."""
    return InvariantIndicators.compute_log_ratio(
        series, reference=reference, window=window, eps=eps
    )


def compute_relative_change(
    series: pd.Series,
    periods: int = 1,
    eps: float = 1e-9,
) -> pd.Series:
    """Wrapper for InvariantIndicators.compute_relative_change."""
    return InvariantIndicators.compute_relative_change(series, periods=periods, eps=eps)


def compute_rolling_energy(
    series: pd.Series,
    window: int = 20,
) -> pd.Series:
    """Wrapper for InvariantIndicators.compute_rolling_energy."""
    return InvariantIndicators.compute_rolling_energy(series, window=window)


def compute_normalized_energy(
    series: pd.Series,
    window: int = 20,
    eps: float = 1e-9,
) -> pd.Series:
    """Wrapper for InvariantIndicators.compute_normalized_energy."""
    return InvariantIndicators.compute_normalized_energy(series, window=window, eps=eps)


def compute_minmax_position(
    series: pd.Series,
    window: int = 20,
    eps: float = 1e-9,
) -> pd.Series:
    """Wrapper for InvariantIndicators.compute_minmax_position."""
    return InvariantIndicators.compute_minmax_position(series, window=window, eps=eps)


def add_invariant_features(
    df: pd.DataFrame,
    columns: list[str],
    window: int = 20,
    prefix: str = "",
    eps: float = 1e-9,
) -> pd.DataFrame:
    """Wrapper for InvariantIndicators.add_invariant_features."""
    return InvariantIndicators.add_invariant_features(
        df, columns=columns, window=window, prefix=prefix, eps=eps
    )


__all__ = [
    "InvariantIndicators",
    "compute_rolling_zscore",
    "compute_robust_zscore",
    "compute_local_ratio",
    "compute_log_ratio",
    "compute_relative_change",
    "compute_rolling_energy",
    "compute_normalized_energy",
    "compute_minmax_position",
    "add_invariant_features",
]
