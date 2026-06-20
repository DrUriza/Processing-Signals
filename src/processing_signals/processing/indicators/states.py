# \file **********************************************************************
# COMPANY:            ELATIN
# PROJECT:            SIGNAL_ANALYSIS
# COMPONENT:          INDICATORS
# MODULE NAME:        states.py
# DESCRIPTION:        @brief Domain-agnostic signal state indicators.
#                     Builds lightweight semantic states from numeric
#                     time-series dynamics without domain-specific coupling.
# CREATION DATE:      06.05.2026
# VERSION:            $Revision: 0.1$
# CHANGES:            06.05.2026 - Initial implementation.
# *****************************************************************************

from __future__ import annotations

import numpy as np
import pandas as pd

from signal_analysis.core.schema import validate_dataframe
from signal_analysis.indicators.variable_action import add_ohlc_geometry_features


class SignalStateIndicators:
    # ***********************************************************************************************************************
    # Functionname:       SignalStateIndicators.compute_slope_state(series, window, eps, flat_threshold)
    #
    # @brief              Compute normalized slope and categorical slope state.
    # @pre                series is array-like; window >= 1.
    # @post               Returns DataFrame with slope, slope_norm, slope_state, slope_confidence.
    # @param[in]          series: Input signal
    #                     window: Rolling period for slope and normalization
    #                     eps: Numerical stability factor
    #                     flat_threshold: Threshold around zero for flat state
    # @param[out]         out: Slope state DataFrame aligned to input index
    # ***********************************************************************************************************************
    @staticmethod
    def compute_slope_state(
        series: pd.Series,
        window: int = 5,
        eps: float = 1e-9,
        flat_threshold: float = 0.001,
    ) -> pd.DataFrame:
        s = pd.Series(series, dtype=float)
        if window < 1:
            raise ValueError("window must be >= 1")

        slope = s.diff(window) / float(window)
        norm_base = s.abs().rolling(window=window).mean()
        slope_norm = slope / (norm_base + eps)

        slope_state = pd.Series("flat", index=s.index, dtype="object")
        slope_state[slope_norm > flat_threshold] = "rising"
        slope_state[slope_norm < -flat_threshold] = "falling"

        raw_conf = (slope_norm.abs() - flat_threshold) / (flat_threshold + eps)
        slope_confidence = raw_conf.clip(lower=0.0, upper=1.0).fillna(0.0)

        return pd.DataFrame(
            {
                "slope": slope,
                "slope_norm": slope_norm,
                "slope_state": slope_state,
                "slope_confidence": slope_confidence,
            },
            index=s.index,
        )

    # ***********************************************************************************************************************
    # Functionname:       SignalStateIndicators.compute_trend_state(df, value_col, fast_col, slow_col,
    #                              window, eps)
    #
    # @brief              Compute generic trend state from value slope and optional fast/slow support.
    # @pre                df is a DataFrame.
    # @post               Returns DataFrame with trend_state and trend_confidence.
    # @param[in]          df: Input DataFrame
    #                     value_col: Primary value column used for baseline slope state
    #                     fast_col: Optional fast signal column
    #                     slow_col: Optional slow signal column
    #                     window: Rolling period
    #                     eps: Numerical stability factor
    # @param[out]         out: Trend state DataFrame
    # ***********************************************************************************************************************
    @staticmethod
    def compute_trend_state(
        df: pd.DataFrame,
        value_col: str = "close",
        fast_col: str | None = None,
        slow_col: str | None = None,
        window: int = 5,
        eps: float = 1e-9,
    ) -> pd.DataFrame:
        validate_dataframe(df)

        if value_col not in df.columns or not pd.api.types.is_numeric_dtype(df[value_col]):
            return pd.DataFrame(
                {
                    "trend_state": pd.Series("sideways", index=df.index, dtype="object"),
                    "trend_confidence": pd.Series(0.0, index=df.index, dtype=float),
                },
                index=df.index,
            )

        slope_df = SignalStateIndicators.compute_slope_state(df[value_col], window=window, eps=eps)
        trend_state = pd.Series("sideways", index=df.index, dtype="object")
        trend_state[slope_df["slope_state"] == "rising"] = "rising"
        trend_state[slope_df["slope_state"] == "falling"] = "falling"

        trend_conf = slope_df["slope_confidence"].copy()

        if (
            fast_col is not None
            and slow_col is not None
            and fast_col in df.columns
            and slow_col in df.columns
            and pd.api.types.is_numeric_dtype(df[fast_col])
            and pd.api.types.is_numeric_dtype(df[slow_col])
        ):
            diff = df[fast_col] - df[slow_col]
            support_rising = diff > 0
            support_falling = diff < 0

            trend_state[(trend_state == "sideways") & support_rising] = "rising"
            trend_state[(trend_state == "sideways") & support_falling] = "falling"

            cross_strength = (diff.abs() / (df[slow_col].abs() + eps)).clip(0.0, 1.0).fillna(0.0)
            trend_conf = np.maximum(trend_conf.to_numpy(dtype=float), cross_strength.to_numpy(dtype=float))
            trend_conf = pd.Series(trend_conf, index=df.index)

        return pd.DataFrame(
            {
                "trend_state": trend_state,
                "trend_confidence": trend_conf.clip(0.0, 1.0),
            },
            index=df.index,
        )

    # ***********************************************************************************************************************
    # Functionname:       SignalStateIndicators.compute_momentum_state(df, momentum_cols, window, eps)
    #
    # @brief              Compute generic momentum state from available momentum-like columns.
    # @pre                df is a DataFrame.
    # @post               Returns momentum_state and momentum_confidence.
    # @param[in]          df: Input DataFrame
    #                     momentum_cols: Optional momentum columns
    #                     window: Rolling period for slope
    #                     eps: Numerical stability factor
    # @param[out]         out: Momentum state DataFrame
    # ***********************************************************************************************************************
    @staticmethod
    def compute_momentum_state(
        df: pd.DataFrame,
        momentum_cols: list[str] | None = None,
        window: int = 5,
        eps: float = 1e-9,
    ) -> pd.DataFrame:
        validate_dataframe(df)

        default_cols = ["rsi", "macd_histogram", "roc", "tsi"]
        candidates = default_cols if momentum_cols is None else momentum_cols
        cols = [
            c for c in candidates
            if c in df.columns and pd.api.types.is_numeric_dtype(df[c])
        ]

        if not cols:
            return pd.DataFrame(
                {
                    "momentum_state": pd.Series("neutral", index=df.index, dtype="object"),
                    "momentum_confidence": pd.Series(0.0, index=df.index, dtype=float),
                },
                index=df.index,
            )

        norm_slopes = []
        for col in cols:
            slope_df = SignalStateIndicators.compute_slope_state(df[col], window=window, eps=eps)
            norm_slopes.append(slope_df["slope_norm"])

        agg = pd.concat(norm_slopes, axis=1).mean(axis=1)
        threshold = 0.01

        state = pd.Series("neutral", index=df.index, dtype="object")
        state[agg > threshold] = "strengthening"
        state[agg < -threshold] = "weakening"

        conf = ((agg.abs() - threshold) / (threshold + eps)).clip(0.0, 1.0).fillna(0.0)

        return pd.DataFrame(
            {
                "momentum_state": state,
                "momentum_confidence": conf,
            },
            index=df.index,
        )

    # ***********************************************************************************************************************
    # Functionname:       SignalStateIndicators.compute_volatility_state(df, volatility_cols, window, eps)
    #
    # @brief              Compute volatility regime state from available volatility proxies.
    # @pre                df is a DataFrame.
    # @post               Returns volatility_state and volatility_confidence.
    # @param[in]          df: Input DataFrame
    #                     volatility_cols: Optional volatility columns
    #                     window: Rolling period
    #                     eps: Numerical stability factor
    # @param[out]         out: Volatility state DataFrame
    # ***********************************************************************************************************************
    @staticmethod
    def compute_volatility_state(
        df: pd.DataFrame,
        volatility_cols: list[str] | None = None,
        window: int = 20,
        eps: float = 1e-9,
    ) -> pd.DataFrame:
        validate_dataframe(df)

        default_cols = ["atr", "bb_width", "range"]
        candidates = default_cols if volatility_cols is None else volatility_cols
        cols = [
            c for c in candidates
            if c in df.columns and pd.api.types.is_numeric_dtype(df[c])
        ]

        if not cols:
            return pd.DataFrame(
                {
                    "volatility_state": pd.Series("normal", index=df.index, dtype="object"),
                    "volatility_confidence": pd.Series(0.0, index=df.index, dtype=float),
                },
                index=df.index,
            )

        proxy = pd.concat([df[c] for c in cols], axis=1).mean(axis=1)
        baseline = proxy.rolling(window=window).median()
        ratio = proxy / (baseline + eps)

        state = pd.Series("normal", index=df.index, dtype="object")
        state[ratio < 0.85] = "compression"
        state[ratio > 1.15] = "expansion"

        conf = ((ratio - 1.0).abs() / 0.15).clip(0.0, 1.0).fillna(0.0)

        return pd.DataFrame(
            {
                "volatility_state": state,
                "volatility_confidence": conf,
            },
            index=df.index,
        )

    # ***********************************************************************************************************************
    # Functionname:       SignalStateIndicators.compute_structure_state(df, eps)
    #
    # @brief              Compute structural candle/segment state from geometry columns.
    # @pre                df is a DataFrame.
    # @post               Returns structure_state and structure_confidence.
    # @param[in]          df: Input DataFrame
    #                     eps: Numerical stability factor
    # @param[out]         out: Structure state DataFrame
    # ***********************************************************************************************************************
    @staticmethod
    def compute_structure_state(
        df: pd.DataFrame,
        eps: float = 1e-9,
    ) -> pd.DataFrame:
        validate_dataframe(df)

        required = ["body_ratio", "upper_wick_ratio", "lower_wick_ratio", "body_direction"]
        local_df = df

        if not all(c in local_df.columns for c in required):
            has_ohlc = all(c in local_df.columns for c in ["open", "high", "low", "close"])
            if has_ohlc:
                local_df = add_ohlc_geometry_features(local_df)
            else:
                return pd.DataFrame(
                    {
                        "structure_state": pd.Series("normal", index=df.index, dtype="object"),
                        "structure_confidence": pd.Series(0.0, index=df.index, dtype=float),
                    },
                    index=df.index,
                )

        body_ratio = pd.Series(local_df["body_ratio"], dtype=float)
        upper = pd.Series(local_df["upper_wick_ratio"], dtype=float)
        lower = pd.Series(local_df["lower_wick_ratio"], dtype=float)
        direction = pd.Series(local_df["body_direction"], dtype=float)

        state = pd.Series("normal", index=local_df.index, dtype="object")

        upper_rej = (upper >= 0.45) & (lower < 0.45)
        lower_rej = (lower >= 0.45) & (upper < 0.45)
        bi_rej = (upper >= 0.45) & (lower >= 0.45)
        strong_dir = (body_ratio >= 0.60) & (direction.abs() > 0)
        indecision = body_ratio <= 0.10

        state[upper_rej] = "upper_rejection"
        state[lower_rej] = "lower_rejection"
        state[bi_rej] = "bidirectional_rejection"
        state[strong_dir] = "strong_directional"
        state[indecision] = "indecision"

        conf = pd.Series(0.3, index=local_df.index, dtype=float)
        conf = np.where(indecision, (0.10 - body_ratio).abs() / (0.10 + eps), conf)
        conf = np.where(strong_dir, body_ratio.clip(0.0, 1.0), conf)
        conf = np.where(upper_rej, upper.clip(0.0, 1.0), conf)
        conf = np.where(lower_rej, lower.clip(0.0, 1.0), conf)
        conf = np.where(bi_rej, np.maximum(upper, lower).clip(0.0, 1.0), conf)
        conf = pd.Series(conf, index=local_df.index).fillna(0.0).clip(0.0, 1.0)

        return pd.DataFrame(
            {
                "structure_state": state,
                "structure_confidence": conf,
            },
            index=local_df.index,
        )

    # ***********************************************************************************************************************
    # Functionname:       SignalStateIndicators.compute_activity_state(df, activity_cols, window, eps)
    #
    # @brief              Compute activity regime from available activity proxy columns.
    # @pre                df is a DataFrame.
    # @post               Returns activity_state and activity_confidence.
    # @param[in]          df: Input DataFrame
    #                     activity_cols: Optional activity columns
    #                     window: Rolling period
    #                     eps: Numerical stability factor
    # @param[out]         out: Activity state DataFrame
    # ***********************************************************************************************************************
    @staticmethod
    def compute_activity_state(
        df: pd.DataFrame,
        activity_cols: list[str] | None = None,
        window: int = 20,
        eps: float = 1e-9,
    ) -> pd.DataFrame:
        validate_dataframe(df)

        default_cols = ["volume", "activity", "energy", "rolling_energy", "normalized_energy"]
        candidates = default_cols if activity_cols is None else activity_cols
        cols = [
            c for c in candidates
            if c in df.columns and pd.api.types.is_numeric_dtype(df[c])
        ]

        if not cols:
            return pd.DataFrame(
                {
                    "activity_state": pd.Series("normal", index=df.index, dtype="object"),
                    "activity_confidence": pd.Series(0.0, index=df.index, dtype=float),
                },
                index=df.index,
            )

        proxy = pd.concat([df[c] for c in cols], axis=1).mean(axis=1)
        baseline = proxy.rolling(window=window).median()
        ratio = proxy / (baseline + eps)

        state = pd.Series("normal", index=df.index, dtype="object")
        state[ratio < 0.75] = "calm"
        state[ratio > 1.25] = "active"
        state[ratio > 2.0] = "extreme"

        conf = ((ratio - 1.0).abs() / 1.0).clip(0.0, 1.0).fillna(0.0)

        return pd.DataFrame(
            {
                "activity_state": state,
                "activity_confidence": conf,
            },
            index=df.index,
        )

    # ***********************************************************************************************************************
    # Functionname:       SignalStateIndicators.build_signal_state_frame(df, value_col, fast_col,
    #                              slow_col, momentum_cols, volatility_cols, activity_cols, window)
    #
    # @brief              Build combined state frame for trend, momentum, volatility, structure, activity.
    # @pre                df is a DataFrame.
    # @post               Returns DataFrame aligned to input index with state/confidence columns.
    # @param[in]          df: Input DataFrame
    #                     value_col: Primary value column for trend
    #                     fast_col: Optional fast support column
    #                     slow_col: Optional slow support column
    #                     momentum_cols: Optional momentum columns
    #                     volatility_cols: Optional volatility columns
    #                     activity_cols: Optional activity columns
    #                     window: Rolling window parameter
    # @param[out]         out: Combined signal state DataFrame
    # ***********************************************************************************************************************
    @staticmethod
    def build_signal_state_frame(
        df: pd.DataFrame,
        value_col: str = "close",
        fast_col: str | None = None,
        slow_col: str | None = None,
        momentum_cols: list[str] | None = None,
        volatility_cols: list[str] | None = None,
        activity_cols: list[str] | None = None,
        window: int = 20,
    ) -> pd.DataFrame:
        validate_dataframe(df)

        trend_df = SignalStateIndicators.compute_trend_state(
            df,
            value_col=value_col,
            fast_col=fast_col,
            slow_col=slow_col,
            window=max(2, min(window, 10)),
        )
        momentum_df = SignalStateIndicators.compute_momentum_state(
            df,
            momentum_cols=momentum_cols,
            window=max(2, min(window, 10)),
        )
        volatility_df = SignalStateIndicators.compute_volatility_state(
            df,
            volatility_cols=volatility_cols,
            window=max(3, window),
        )
        structure_df = SignalStateIndicators.compute_structure_state(df)
        activity_df = SignalStateIndicators.compute_activity_state(
            df,
            activity_cols=activity_cols,
            window=max(3, window),
        )

        out = pd.concat([trend_df, momentum_df, volatility_df, structure_df, activity_df], axis=1)
        out.index = df.index
        return out


# ---------------------------------------------------------------------------
# Module-level convenience wrappers
# ---------------------------------------------------------------------------


def compute_slope_state(
    series: pd.Series,
    window: int = 5,
    eps: float = 1e-9,
    flat_threshold: float = 0.001,
) -> pd.DataFrame:
    return SignalStateIndicators.compute_slope_state(
        series,
        window=window,
        eps=eps,
        flat_threshold=flat_threshold,
    )


def compute_trend_state(
    df: pd.DataFrame,
    value_col: str = "close",
    fast_col: str | None = None,
    slow_col: str | None = None,
    window: int = 5,
    eps: float = 1e-9,
) -> pd.DataFrame:
    return SignalStateIndicators.compute_trend_state(
        df,
        value_col=value_col,
        fast_col=fast_col,
        slow_col=slow_col,
        window=window,
        eps=eps,
    )


def compute_momentum_state(
    df: pd.DataFrame,
    momentum_cols: list[str] | None = None,
    window: int = 5,
    eps: float = 1e-9,
) -> pd.DataFrame:
    return SignalStateIndicators.compute_momentum_state(
        df,
        momentum_cols=momentum_cols,
        window=window,
        eps=eps,
    )


def compute_volatility_state(
    df: pd.DataFrame,
    volatility_cols: list[str] | None = None,
    window: int = 20,
    eps: float = 1e-9,
) -> pd.DataFrame:
    return SignalStateIndicators.compute_volatility_state(
        df,
        volatility_cols=volatility_cols,
        window=window,
        eps=eps,
    )


def compute_structure_state(
    df: pd.DataFrame,
    eps: float = 1e-9,
) -> pd.DataFrame:
    return SignalStateIndicators.compute_structure_state(df, eps=eps)


def compute_activity_state(
    df: pd.DataFrame,
    activity_cols: list[str] | None = None,
    window: int = 20,
    eps: float = 1e-9,
) -> pd.DataFrame:
    return SignalStateIndicators.compute_activity_state(
        df,
        activity_cols=activity_cols,
        window=window,
        eps=eps,
    )


def build_signal_state_frame(
    df: pd.DataFrame,
    value_col: str = "close",
    fast_col: str | None = None,
    slow_col: str | None = None,
    momentum_cols: list[str] | None = None,
    volatility_cols: list[str] | None = None,
    activity_cols: list[str] | None = None,
    window: int = 20,
) -> pd.DataFrame:
    return SignalStateIndicators.build_signal_state_frame(
        df,
        value_col=value_col,
        fast_col=fast_col,
        slow_col=slow_col,
        momentum_cols=momentum_cols,
        volatility_cols=volatility_cols,
        activity_cols=activity_cols,
        window=window,
    )


__all__ = [
    "SignalStateIndicators",
    "compute_slope_state",
    "compute_trend_state",
    "compute_momentum_state",
    "compute_volatility_state",
    "compute_structure_state",
    "compute_activity_state",
    "build_signal_state_frame",
]
