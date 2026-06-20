# \file **********************************************************************
# COMPANY:            ELATIN
# PROJECT:            SIGNAL_ANALYSIS
# COMPONENT:          CORE
# MODULE NAME:        sync.py
# DESCRIPTION:        @brief Temporal multi-channel signal synchronization utilities.
#                     Align, validate, and diagnose temporal alignment of multiple
#                     signal channels with different latency or sampling rates.
# CREATION DATE:      06.05.2026
# VERSION:            $Revision: 0.1$
# CHANGES:            06.05.2026 - Initial implementation.
# *****************************************************************************

from __future__ import annotations

import numpy as np
import pandas as pd

from signal_analysis.core.schema import validate_dataframe


class TemporalSync:
    """
    Multi-channel temporal signal synchronization.

    Groups alignment, coverage diagnostics, and channel merging into
    a single domain-agnostic class. All methods operate on generic
    pandas DataFrames or Series indexed by any comparable index type.
    """

    # ***********************************************************************************************************************
    # Functionname:       TemporalSync.align_to_index(df, reference_index, method, fill_limit)
    #
    # @brief              Align a DataFrame to a reference index by reindexing and filling.
    # @pre                df is a valid DataFrame; method is a valid pandas fill method.
    # @post               Returns DataFrame with reference_index as index; gaps filled by method.
    # @param[in]          df: Input DataFrame to align
    #                     reference_index: Target pandas Index to align to
    #                     method: Fill method for gaps ('ffill', 'bfill', None)
    #                     fill_limit: Maximum consecutive fill steps; None = unlimited
    # @param[out]         out: Aligned DataFrame
    #
    # @callsequence       @startuml
    #                     title align_to_index
    #                     start
    #                       :validate_dataframe(df);
    #                       :df.reindex(reference_index, method, limit);
    #                       :return aligned DataFrame;
    #                     end
    #                     @enduml
    # ***********************************************************************************************************************
    @staticmethod
    def align_to_index(
        df: pd.DataFrame,
        reference_index: pd.Index,
        method: str | None = "ffill",
        fill_limit: int | None = None,
    ) -> pd.DataFrame:
        validate_dataframe(df)
        return df.reindex(reference_index, method=method, limit=fill_limit)

    # ***********************************************************************************************************************
    # Functionname:       TemporalSync.align_channels(channels, reference_key, method, fill_limit)
    #
    # @brief              Align multiple named channels to a common reference channel index.
    # @pre                channels is a non-empty dict of str->DataFrame; reference_key exists.
    # @post               Returns dict with all channels reindexed to the reference channel's index.
    # @param[in]          channels: Mapping of channel name to DataFrame
    #                     reference_key: Key of the channel used as index reference
    #                     method: Pandas fill method for gaps
    #                     fill_limit: Maximum consecutive fill steps
    # @param[out]         out: Dict of aligned DataFrames, same keys as input
    #
    # @callsequence       @startuml
    #                     title align_channels
    #                     start
    #                     :validate channels not empty;
    #                     :validate reference_key exists;
    #                     :extract ref_index from channels[reference_key];
    #                     repeat
    #                       :align_to_index(df, ref_index, method, fill_limit);
    #                     repeat while (more channels?)
    #                     :return aligned dict;
    #                     end
    #                     @enduml
    # ***********************************************************************************************************************
    @staticmethod
    def align_channels(
        channels: dict[str, pd.DataFrame],
        reference_key: str,
        method: str | None = "ffill",
        fill_limit: int | None = None,
    ) -> dict[str, pd.DataFrame]:
        if not channels:
            raise ValueError("channels must not be empty.")
        if reference_key not in channels:
            raise ValueError(
                f"reference_key '{reference_key}' not found in channels. "
                f"Available: {sorted(channels.keys())}"
            )

        ref_index = channels[reference_key].index
        aligned = {}

        for key, df in channels.items():
            validate_dataframe(df)
            aligned[key] = TemporalSync.align_to_index(
                df=df,
                reference_index=ref_index,
                method=method,
                fill_limit=fill_limit,
            )

        return aligned

    # ***********************************************************************************************************************
    # Functionname:       TemporalSync.compute_coverage_ratio(series, reference_index)
    #
    # @brief              Compute fraction of reference index positions with non-NaN values.
    # @pre                series is Series-like; reference_index is a valid pandas Index.
    # @post               Returns float in [0.0, 1.0].
    # @param[in]          series: Input signal to evaluate
    #                     reference_index: Reference index to project against
    # @param[out]         out: Coverage ratio (non-NaN count / len(reference_index))
    #
    # @callsequence       @startuml
    #                     title compute_coverage_ratio
    #                     start
    #                     :series.reindex(reference_index);
    #                     if (len(reference_index) == 0?) then (yes)
    #                       :return 0.0;
    #                     else (no)
    #                       :return notna().sum() / len(reference_index);
    #                     endif
    #                     end
    #                     @enduml
    # ***********************************************************************************************************************
    @staticmethod
    def compute_coverage_ratio(
        series: pd.Series,
        reference_index: pd.Index,
    ) -> float:
        s = pd.Series(series, dtype=float)
        aligned = s.reindex(reference_index)
        n_ref = len(reference_index)
        if n_ref == 0:
            return 0.0
        return float(aligned.notna().sum()) / n_ref

    # ***********************************************************************************************************************
    # Functionname:       TemporalSync.compute_missing_ratio(series)
    #
    # @brief              Compute fraction of NaN values in a series.
    # @pre                series is array-like (empty allowed).
    # @post               Returns float in [0.0, 1.0]; 0.0 for empty series.
    # @param[in]          series: Input signal
    # @param[out]         out: Missing ratio (NaN count / len(series))
    #
    # @callsequence       @startuml
    #                     title compute_missing_ratio
    #                     start
    #                     if (len(series) == 0?) then (yes)
    #                       :return 0.0;
    #                     else (no)
    #                       :return series.isna().mean();
    #                     endif
    #                     end
    #                     @enduml
    # ***********************************************************************************************************************
    @staticmethod
    def compute_missing_ratio(series: pd.Series) -> float:
        s = pd.Series(series, dtype=float)
        if len(s) == 0:
            return 0.0
        return float(s.isna().mean())

    # ***********************************************************************************************************************
    # Functionname:       TemporalSync.diagnose_alignment(channels, reference_key)
    #
    # @brief              Compute per-channel alignment diagnostics against a reference.
    # @pre                channels is a non-empty dict of str->DataFrame/Series; reference_key exists.
    # @post               Returns DataFrame indexed by channel name with coverage_ratio,
    #                     missing_ratio, and n_samples columns.
    # @param[in]          channels: Mapping of channel name to DataFrame or Series
    #                     reference_key: Key of the reference channel
    # @param[out]         out: Diagnostics DataFrame (rows = channels, cols = metrics)
    #
    # @callsequence       @startuml
    #                     title diagnose_alignment
    #                     start
    #                     :validate channels not empty;
    #                     :validate reference_key exists;
    #                     :extract ref_index;
    #                     repeat
    #                       if (channel is DataFrame?) then (yes)
    #                         :compute_coverage_ratio on first numeric col;
    #                       else (no, Series)
    #                         :compute_coverage_ratio on series;
    #                       endif
    #                       :compute_missing_ratio;
    #                       :append row {channel, cov, miss, n};
    #                     repeat while (more channels?)
    #                     :return pd.DataFrame(rows).set_index("channel");
    #                     end
    #                     @enduml
    # ***********************************************************************************************************************
    @staticmethod
    def diagnose_alignment(
        channels: dict[str, pd.DataFrame | pd.Series],
        reference_key: str,
    ) -> pd.DataFrame:
        if not channels:
            raise ValueError("channels must not be empty.")
        if reference_key not in channels:
            raise ValueError(
                f"reference_key '{reference_key}' not found in channels. "
                f"Available: {sorted(channels.keys())}"
            )

        ref = channels[reference_key]
        ref_index = ref.index if hasattr(ref, "index") else pd.RangeIndex(len(ref))

        rows = []
        for key, channel in channels.items():
            if isinstance(channel, pd.DataFrame):
                numeric_cols = channel.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) == 0:
                    cov, miss, n = 0.0, 1.0, len(channel)
                else:
                    col_series = channel[numeric_cols[0]]
                    cov = TemporalSync.compute_coverage_ratio(col_series, ref_index)
                    miss = TemporalSync.compute_missing_ratio(col_series)
                    n = len(channel)
            elif isinstance(channel, pd.Series):
                cov = TemporalSync.compute_coverage_ratio(channel, ref_index)
                miss = TemporalSync.compute_missing_ratio(channel)
                n = len(channel)
            else:
                cov, miss, n = 0.0, 1.0, 0

            rows.append(
                {"channel": key, "coverage_ratio": cov, "missing_ratio": miss, "n_samples": n}
            )

        return pd.DataFrame(rows).set_index("channel")

    # ***********************************************************************************************************************
    # Functionname:       TemporalSync.merge_channels(channels, reference_key, method, fill_limit)
    #
    # @brief              Align and merge all channels into a single wide DataFrame.
    # @pre                channels is a non-empty dict of named DataFrames; reference_key exists.
    # @post               Returns wide DataFrame with columns prefixed as {channel_name}_{col}.
    # @param[in]          channels: Mapping of channel name to DataFrame
    #                     reference_key: Key used as alignment reference index
    #                     method: Fill method for temporal alignment
    #                     fill_limit: Maximum consecutive fill steps
    # @param[out]         out: Merged wide DataFrame aligned to reference index
    #
    # @callsequence       @startuml
    #                     title merge_channels
    #                     start
    #                     :align_channels(channels, reference_key, method, fill_limit);
    #                     repeat
    #                       :prefix column names as {channel_name}_{col};
    #                       :append to parts list;
    #                     repeat while (more aligned channels?)
    #                     :pd.concat(parts, axis=1);
    #                     :return merged DataFrame;
    #                     end
    #                     @enduml
    # ***********************************************************************************************************************
    @staticmethod
    def merge_channels(
        channels: dict[str, pd.DataFrame],
        reference_key: str,
        method: str | None = "ffill",
        fill_limit: int | None = None,
    ) -> pd.DataFrame:
        aligned = TemporalSync.align_channels(
            channels=channels,
            reference_key=reference_key,
            method=method,
            fill_limit=fill_limit,
        )

        parts = []
        for key, df in aligned.items():
            renamed = df.copy()
            renamed.columns = [f"{key}_{col}" for col in df.columns]
            parts.append(renamed)

        return pd.concat(parts, axis=1)


# ---------------------------------------------------------------------------
# Module-level convenience wrappers
# ---------------------------------------------------------------------------


def align_to_index(
    df: pd.DataFrame,
    reference_index: pd.Index,
    method: str | None = "ffill",
    fill_limit: int | None = None,
) -> pd.DataFrame:
    """Wrapper for TemporalSync.align_to_index."""
    return TemporalSync.align_to_index(
        df, reference_index=reference_index, method=method, fill_limit=fill_limit
    )


def align_channels(
    channels: dict[str, pd.DataFrame],
    reference_key: str,
    method: str | None = "ffill",
    fill_limit: int | None = None,
) -> dict[str, pd.DataFrame]:
    """Wrapper for TemporalSync.align_channels."""
    return TemporalSync.align_channels(
        channels, reference_key=reference_key, method=method, fill_limit=fill_limit
    )


def compute_coverage_ratio(
    series: pd.Series,
    reference_index: pd.Index,
) -> float:
    """Wrapper for TemporalSync.compute_coverage_ratio."""
    return TemporalSync.compute_coverage_ratio(series, reference_index)


def compute_missing_ratio(series: pd.Series) -> float:
    """Wrapper for TemporalSync.compute_missing_ratio."""
    return TemporalSync.compute_missing_ratio(series)


def diagnose_alignment(
    channels: dict[str, pd.DataFrame | pd.Series],
    reference_key: str,
) -> pd.DataFrame:
    """Wrapper for TemporalSync.diagnose_alignment."""
    return TemporalSync.diagnose_alignment(channels, reference_key=reference_key)


def merge_channels(
    channels: dict[str, pd.DataFrame],
    reference_key: str,
    method: str | None = "ffill",
    fill_limit: int | None = None,
) -> pd.DataFrame:
    """Wrapper for TemporalSync.merge_channels."""
    return TemporalSync.merge_channels(
        channels, reference_key=reference_key, method=method, fill_limit=fill_limit
    )


__all__ = [
    "TemporalSync",
    "align_to_index",
    "align_channels",
    "compute_coverage_ratio",
    "compute_missing_ratio",
    "diagnose_alignment",
    "merge_channels",
]
