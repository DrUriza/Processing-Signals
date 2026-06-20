from __future__ import annotations
import numpy    as np
import pandas   as pd

from signal_analysis.utils.ohlc import typical_price

# \file **********************************************************************
# COMPANY:            ELATIN
# PROJECT:            SIGNAL_ANALYSIS
# COMPONENT:          INDICATORS
# MODULE NAME:        volume.py
# DESCRIPTION:        @brief Volume indicators and volume-flow feature builders
# CREATION DATE:      22.04.2026
# VERSION:            $Revision: 0.4$
# CHANGES:            22.04.2026 - Refactored into grouped volume feature blocks.
#                     23.04.2026 - Consolidated to 4 public methods.
# ****************************************************************************

class VolumeIndicators:
    # *******************************************************************************************************************
    # Functionname:       VolumeIndicators._validate_volume_inputs(df: pd.DataFrame,
    #                              require_flow: bool = False)
    #
    # @brief              Validate required OHLCV and optional order-flow columns.
    # @pre                df should be a pandas DataFrame.
    # @post               Raises ValueError when required schema is missing.
    # @param[in]          df: Input market DataFrame
    #                     require_flow: Require buy_volume and sell_volume columns when True
    # @param[out]         out: None
    #
    # @callsequence       @startuml
    #                     title VolumeIndicators._validate_volume_inputs
    #
    #                     start
    #                     :Check df is a pandas DataFrame;
    #                     if (not isinstance?) then (yes)
    #                       :Raise ValueError;
    #                       stop
    #                     endif
    #                     :Check required_cols subset of df.columns;
    #                     if (missing cols?) then (yes)
    #                       :Raise ValueError;
    #                       stop
    #                     endif
    #                     if (require_flow == True?) then (yes)
    #                       :Check buy_volume and sell_volume in df.columns;
    #                       if (missing flow cols?) then (yes)
    #                         :Raise ValueError;
    #                         stop
    #                       endif
    #                     endif
    #                     end
    #                     @enduml
    #
    # @InOutCorrelation   As described in UML diagram
    # @traceability
    # *******************************************************************************************************************
    @staticmethod
    def _validate_volume_inputs(df: pd.DataFrame,
                                require_flow: bool = False) -> None:
        required_cols = {"open", "high", "low", "close", "volume"}
        if not isinstance(df, pd.DataFrame):
            raise ValueError("df must be a pandas DataFrame.")
        if not required_cols.issubset(df.columns):
            raise ValueError("df must contain: open, high, low, close, volume.")
        if require_flow:
            flow_cols = {"buy_volume", "sell_volume"}
            if not flow_cols.issubset(df.columns):
                raise ValueError("df must contain buy_volume and sell_volume when require_flow=True.")

    # *******************************************************************************************************************
    # Functionname:       VolumeIndicators.compute_volume_stats(df: pd.DataFrame,
    #                              window: int = 20, ema_span: int | None = None,
    #                              eps: float = 1e-9)
    #
    # @brief              Compute core rolling volume statistics and OBV in one block.
    #                     Produces: vol_sma, vol_ema, rvol, vol_zscore, obv.
    # @pre                df includes volume, close.
    # @post               Returns DataFrame with all volume statistic columns.
    # @param[in]          df: Input market DataFrame
    #                     window: Rolling window length
    #                     ema_span: Optional EMA span override
    #                     eps: Numerical stabilization factor
    # @param[out]         out: Volume statistics DataFrame
    #
    # @callsequence       @startuml
    #                     title VolumeIndicators.compute_volume_stats
    #
    #                     start
    #                     :Call _validate_volume_inputs(df);
    #                     :Resolve span = ema_span if set, else window;
    #                     :Compute rolling mean_vol over window;
    #                     :Compute rolling std_vol over window;
    #                     :Compute EMA ema_vol with span;
    #                     :Compute rvol = volume / (mean_vol + eps);
    #                     :Compute zscore = (volume - mean_vol) / (std_vol + eps);
    #                     :Compute direction = sign(close.diff()), fillna(0);
    #                     :Compute obv = cumsum(direction * volume);
    #                     :Return DataFrame with vol_sma, vol_ema,
    #                     rvol, vol_zscore, obv;
    #                     end
    #                     @enduml
    #
    # @InOutCorrelation   As described in UML diagram
    # @traceability
    # *******************************************************************************************************************
    @staticmethod
    def compute_volume_stats(df: pd.DataFrame,
                             window: int = 20,
                             ema_span: int | None = None,
                             eps: float = 1e-9) -> pd.DataFrame:
        VolumeIndicators._validate_volume_inputs(df=df)

        span      = window if ema_span is None else ema_span
        mean_vol  = df["volume"].rolling(window=window, min_periods=window).mean()
        std_vol   = df["volume"].rolling(window=window, min_periods=window).std()
        ema_vol   = df["volume"].ewm(span=span, adjust=False).mean()
        rvol      = df["volume"] / (mean_vol + eps)
        zscore    = (df["volume"] - mean_vol) / (std_vol + eps)
        direction = np.sign(df["close"].diff()).fillna(0.0)
        obv       = (direction * df["volume"]).cumsum()

        return pd.DataFrame({
            f"vol_sma_{window}":    mean_vol,
            f"vol_ema_{span}":      ema_vol,
            f"rvol_{window}":       rvol,
            f"vol_zscore_{window}": zscore,
            "obv":                  obv,
        }, index=df.index)

    # *******************************************************************************************************************
    # Functionname:       VolumeIndicators.compute_vwap_metrics(df: pd.DataFrame,
    #                              rolling_window: int = 20, mfi_window: int = 14,
    #                              cmf_window: int = 20, eps: float = 1e-9)
    #
    # @brief              Compute price-volume metrics in one block: typical price,
    #                     cumulative VWAP, rolling VWAP, close-to-VWAP distance,
    #                     Money Flow Index (MFI), and Chaikin Money Flow (CMF).
    # @pre                df includes OHLCV.
    # @post               Returns DataFrame with all price-volume columns.
    # @param[in]          df: Input market DataFrame
    #                     rolling_window: Window length for rolling VWAP
    #                     mfi_window: Window length for Money Flow Index
    #                     cmf_window: Window length for Chaikin Money Flow
    #                     eps: Numerical stabilization factor
    # @param[out]         out: Price-volume metrics DataFrame
    #
    # @callsequence       @startuml
    #                     title VolumeIndicators.compute_vwap_metrics
    #
    #                     start
    #                     :Call _validate_volume_inputs(df);
    #                     :Call typical_price(high, low, close) -> tp;
    #                     :Compute tpv = tp * volume;
    #                     :Compute cumulative vwap = cumsum(tpv) / cumsum(volume);
    #                     :Compute rolling_vwap over rolling_window;
    #                     :Compute close_vwap_dist = (close - vwap) / (vwap + eps);
    #                     :Compute tp_diff = tp.diff();
    #                     :Split raw_money_flow into positive_flow / negative_flow;
    #                     :Roll sums over mfi_window;
    #                     :Compute MFI = 100 - 100 / (1 + pos_sum / neg_sum);
    #                     :Compute money_flow_multiplier from HL range;
    #                     :Compute CMF over cmf_window;
    #                     :Return DataFrame with typical_price, vwap,
    #                     rolling_vwap, close_vwap_dist, mfi, cmf;
    #                     end
    #                     @enduml
    #
    # @InOutCorrelation   As described in UML diagram
    # @traceability
    # *******************************************************************************************************************
    @staticmethod
    def compute_vwap_metrics(df: pd.DataFrame,
                             rolling_window: int = 20,
                             mfi_window: int = 14,
                             cmf_window: int = 20,
                             eps: float = 1e-9) -> pd.DataFrame:
        VolumeIndicators._validate_volume_inputs(df=df)

        tp              = typical_price(df["high"], df["low"], df["close"])
        tpv             = tp * df["volume"]
        vwap_series     = tpv.cumsum() / (df["volume"].cumsum() + eps)
        rolling_vwap    = (tpv.rolling(window=rolling_window, min_periods=rolling_window).sum()
                           / (df["volume"].rolling(window=rolling_window, min_periods=rolling_window).sum() + eps))
        close_vwap_dist = (df["close"] - vwap_series) / (vwap_series + eps)

        raw_mf   = tpv
        tp_diff  = tp.diff()
        pos_flow = raw_mf.where(tp_diff > 0.0, 0.0)
        neg_flow = raw_mf.where(tp_diff < 0.0, 0.0).abs()
        pos_sum  = pos_flow.rolling(window=mfi_window, min_periods=mfi_window).sum()
        neg_sum  = neg_flow.rolling(window=mfi_window, min_periods=mfi_window).sum()
        mfi      = 100.0 - (100.0 / (1.0 + pos_sum / (neg_sum + eps)))

        hl_range = (df["high"] - df["low"]).replace(0.0, np.nan)
        mfm      = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / (hl_range + eps)
        cmf      = ((mfm * df["volume"]).rolling(window=cmf_window, min_periods=cmf_window).sum()
                    / (df["volume"].rolling(window=cmf_window, min_periods=cmf_window).sum() + eps))

        return pd.DataFrame({
            "typical_price":                  tp,
            "vwap":                           vwap_series,
            f"rolling_vwap_{rolling_window}": rolling_vwap,
            "close_vwap_dist":                close_vwap_dist,
            f"mfi_{mfi_window}":              mfi,
            f"cmf_{cmf_window}":              cmf,
        }, index=df.index)

    # *******************************************************************************************************************
    # Functionname:       VolumeIndicators.compute_order_flow_metrics(df: pd.DataFrame,
    #                              rvol_col: str | None = None,
    #                              atr_col: str | None = None,
    #                              body_weight: float = 1.0,
    #                              volume_weight: float = 1.0,
    #                              eps: float = 1e-9)
    #
    # @brief              Compute order-flow and breakout metrics in one block.
    #                     When buy_volume / sell_volume are present: bar delta,
    #                     normalized imbalance, cumulative delta.
    #                     When rvol_col is supplied: composite breakout score.
    # @pre                df includes OHLCV; buy/sell columns optional.
    # @post               Returns DataFrame with available order-flow columns.
    # @param[in]          df: Input market DataFrame
    #                     rvol_col: Relative-volume column name used in breakout scoring
    #                     atr_col: ATR column used to normalize candle body
    #                     body_weight: Weight of normalized body component
    #                     volume_weight: Weight of relative-volume component
    #                     eps: Numerical stabilization factor
    # @param[out]         out: Order-flow metrics DataFrame
    #
    # @callsequence       @startuml
    #                     title VolumeIndicators.compute_order_flow_metrics
    #
    #                     start
    #                     :Call _validate_volume_inputs(df);
    #                     :Initialize empty result dict;
    #                     if (buy_volume and sell_volume in df?) then (yes)
    #                       :Compute delta = buy_volume - sell_volume;
    #                       :Compute imbalance = delta / (total + eps);
    #                       :Compute cvd = delta.cumsum();
    #                       :Add volume_delta, volume_imbalance, cvd to result;
    #                     endif
    #                     if (rvol_col provided and in df?) then (yes)
    #                       :Compute body = abs(close - open);
    #                       if (atr_col provided and in df?) then (yes)
    #                         :Normalize body by df[atr_col];
    #                       else (no)
    #                         :Normalize body by (high - low);
    #                       endif
    #                       :Compute score = body_weight * body_norm
    #                       + volume_weight * df[rvol_col];
    #                       :Add breakout_volume_score to result;
    #                     endif
    #                     :Return DataFrame from result dict;
    #                     end
    #                     @enduml
    #
    # @InOutCorrelation   As described in UML diagram
    # @traceability
    # *******************************************************************************************************************
    @staticmethod
    def compute_order_flow_metrics(df: pd.DataFrame,
                                   rvol_col: str | None = None,
                                   atr_col: str | None = None,
                                   body_weight: float = 1.0,
                                   volume_weight: float = 1.0,
                                   eps: float = 1e-9) -> pd.DataFrame:
        VolumeIndicators._validate_volume_inputs(df=df)

        result: dict[str, pd.Series] = {}

        if {"buy_volume", "sell_volume"}.issubset(df.columns):
            delta                      = df["buy_volume"] - df["sell_volume"]
            total                      = df["buy_volume"] + df["sell_volume"]
            result["volume_delta"]     = delta
            result["volume_imbalance"] = delta / (total + eps)
            result["cvd"]              = delta.cumsum()

        if rvol_col is not None and rvol_col in df.columns:
            body      = (df["close"] - df["open"]).abs()
            hl_range  = (df["high"] - df["low"]).replace(0.0, np.nan)
            norm_denom = df[atr_col].replace(0.0, np.nan) if atr_col and atr_col in df.columns else hl_range
            body_norm  = body / norm_denom
            result["breakout_volume_score"] = (body_weight * body_norm) + (volume_weight * df[rvol_col])

        return pd.DataFrame(result, index=df.index)

    # *******************************************************************************************************************
    # Functionname:       VolumeIndicators.add_volume_features(df: pd.DataFrame,
    #                              vol_window: int = 20, ema_span: int | None = None,
    #                              rolling_window: int = 20, mfi_window: int = 14,
    #                              cmf_window: int = 20, flow_enabled: bool = True,
    #                              atr_col: str | None = None, eps: float = 1e-9)
    #
    # @brief              Build the full volume feature pack by calling all
    #                     grouped metric blocks and concatenating into one DataFrame.
    # @pre                df includes OHLCV; buy/sell columns optional.
    # @post               Returns enriched DataFrame copy.
    # @param[in]          df: Input market DataFrame
    #                     vol_window: Window length for rolling volume statistics
    #                     ema_span: Optional EMA span override
    #                     rolling_window: Window length for rolling VWAP
    #                     mfi_window: Window length for Money Flow Index
    #                     cmf_window: Window length for Chaikin Money Flow
    #                     flow_enabled: Include flow metrics when True
    #                     atr_col: ATR column used in breakout scoring
    #                     eps: Numerical stabilization factor
    # @param[out]         out: Enriched DataFrame
    #
    # @callsequence       @startuml
    #                     title VolumeIndicators.add_volume_features
    #
    #                     start
    #                     :Call _validate_volume_inputs(df);
    #                     :Copy df -> out;
    #                     :Call compute_volume_stats(out, vol_window, ema_span, eps);
    #                     :Concat volume stats into out;
    #                     :Call compute_vwap_metrics(out, rolling_window,
    #                     mfi_window, cmf_window, eps);
    #                     :Concat vwap/flow metrics into out;
    #                     if (flow_enabled?) then (yes)
    #                       :Call compute_order_flow_metrics(out,
    #                       rvol_col=rvol_{vol_window}, atr_col, ...);
    #                       if (result not empty?) then (yes)
    #                         :Concat order-flow metrics into out;
    #                       endif
    #                     endif
    #                     :Return out;
    #                     end
    #                     @enduml
    #
    # @InOutCorrelation   As described in UML diagram
    # @traceability
    # *******************************************************************************************************************
    @staticmethod
    def add_volume_features(df: pd.DataFrame,
                            vol_window: int = 20,
                            ema_span: int | None = None,
                            rolling_window: int = 20,
                            mfi_window: int = 14,
                            cmf_window: int = 20,
                            flow_enabled: bool = True,
                            atr_col: str | None = None,
                            eps: float = 1e-9) -> pd.DataFrame:
        VolumeIndicators._validate_volume_inputs(df=df)

        out        = df.copy()
        vol_stats  = VolumeIndicators.compute_volume_stats(df=out, window=vol_window,
                                                           ema_span=ema_span, eps=eps)
        out        = pd.concat([out, vol_stats], axis=1)

        vwap_block = VolumeIndicators.compute_vwap_metrics(df=out, rolling_window=rolling_window,
                                                           mfi_window=mfi_window, cmf_window=cmf_window,
                                                           eps=eps)
        out = pd.concat([out, vwap_block], axis=1)

        if flow_enabled:
            rvol_col   = f"rvol_{vol_window}"
            flow_block = VolumeIndicators.compute_order_flow_metrics(df=out, rvol_col=rvol_col,
                                                                     atr_col=atr_col, eps=eps)
            if not flow_block.empty:
                out = pd.concat([out, flow_block], axis=1)

        return out


# ---------------------------------------------------------------------------
# Module-level convenience aliases
# ---------------------------------------------------------------------------

def compute_volume_stats(df: pd.DataFrame,
                         window: int = 20,
                         ema_span: int | None = None,
                         eps: float = 1e-9) -> pd.DataFrame:
    return VolumeIndicators.compute_volume_stats(df=df, window=window, ema_span=ema_span, eps=eps)


def compute_vwap_metrics(df: pd.DataFrame,
                         rolling_window: int = 20,
                         mfi_window: int = 14,
                         cmf_window: int = 20,
                         eps: float = 1e-9) -> pd.DataFrame:
    return VolumeIndicators.compute_vwap_metrics(df=df, rolling_window=rolling_window,
                                                 mfi_window=mfi_window, cmf_window=cmf_window, eps=eps)


def compute_order_flow_metrics(df: pd.DataFrame,
                               rvol_col: str | None = None,
                               atr_col: str | None = None,
                               body_weight: float = 1.0,
                               volume_weight: float = 1.0,
                               eps: float = 1e-9) -> pd.DataFrame:
    return VolumeIndicators.compute_order_flow_metrics(df=df, rvol_col=rvol_col, atr_col=atr_col,
                                                       body_weight=body_weight, volume_weight=volume_weight,
                                                       eps=eps)


def add_volume_features(df: pd.DataFrame,
                        vol_window: int = 20,
                        ema_span: int | None = None,
                        rolling_window: int = 20,
                        mfi_window: int = 14,
                        cmf_window: int = 20,
                        flow_enabled: bool = True,
                        atr_col: str | None = None,
                        eps: float = 1e-9) -> pd.DataFrame:
    return VolumeIndicators.add_volume_features(df=df, vol_window=vol_window, ema_span=ema_span,
                                                rolling_window=rolling_window, mfi_window=mfi_window,
                                                cmf_window=cmf_window, flow_enabled=flow_enabled,
                                                atr_col=atr_col, eps=eps)


__all__ = [
    "VolumeIndicators",
    "compute_volume_stats",
    "compute_vwap_metrics",
    "compute_order_flow_metrics",
    "add_volume_features",
]
