"""Core schema, feature window, and temporal sync module."""

from signal_analysis.core.schema import (
    OPTIONAL_SIGNED_FLOW_COLUMNS,
    OPTIONAL_TIME_COLUMNS,
    OPTIONAL_VOLUME_COLUMNS,
    QUALITY_COLUMNS,
    REQUIRED_OHLC_COLUMNS,
    build_feature_contract,
    infer_time_column,
    validate_dataframe,
    validate_numeric_columns,
    validate_ohlc_columns,
    validate_required_columns,
)
from signal_analysis.core.feature_window import (
    WINDOW_META_KEYS,
    WINDOW_SECTION_KEYS,
    build_empty_window,
    build_window_meta,
    build_windowed_sequence,
    extract_window_from_dataframe,
    validate_window,
)
from signal_analysis.core.sync import (
    TemporalSync,
    align_channels,
    align_to_index,
    compute_coverage_ratio,
    compute_missing_ratio,
    diagnose_alignment,
    merge_channels,
)
from signal_analysis.core.window_builder import (
    WindowBuilder,
    build_feature_sections,
    build_window,
    build_window_sequence,
    flatten_window,
    summarize_frame,
    windows_to_feature_frame,
)
from signal_analysis.core.feature_pipeline import (
    FeaturePipeline,
    build_feature_matrix,
    build_pipeline_report,
    drop_non_numeric_features,
    fill_feature_matrix,
    validate_feature_matrix,
)

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
    # feature_window
    "WINDOW_SECTION_KEYS",
    "WINDOW_META_KEYS",
    "build_window_meta",
    "build_empty_window",
    "extract_window_from_dataframe",
    "build_windowed_sequence",
    "validate_window",
    # sync
    "TemporalSync",
    "align_to_index",
    "align_channels",
    "compute_coverage_ratio",
    "compute_missing_ratio",
    "diagnose_alignment",
    "merge_channels",
    # window_builder
    "WindowBuilder",
    "summarize_frame",
    "build_feature_sections",
    "build_window",
    "build_window_sequence",
    "flatten_window",
    "windows_to_feature_frame",
    # feature_pipeline
    "FeaturePipeline",
    "build_feature_matrix",
    "validate_feature_matrix",
    "drop_non_numeric_features",
    "fill_feature_matrix",
    "build_pipeline_report",
]
