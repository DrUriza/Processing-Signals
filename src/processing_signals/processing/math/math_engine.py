from __future__ import annotations

from typing import Any

import pandas as pd

from processing_signals.processing.math.statistics import (
    rolling_distribution_features,
    safe_returns,
    summarize_series,
    last_valid_dict,
)
from processing_signals.processing.math.technical_indicators import compute_ohlcv_indicators
from processing_signals.processing.math.microstructure import orderbook_metrics, event_flow_metrics, wall_score_from_orderbook


class ProcessingMathEngine:
    """
    Processing/Math layer.

    Calculates:
      - technical indicators for OHLCV
      - pure statistical metrics for time-series and events
      - microstructure metrics for order book, large trades, and whale orders
    """

    DEFAULT_WINDOWS = [14, 30, 60]

    def compute(self, normalized: dict[str, Any], decision: dict[str, Any]) -> dict[str, Any]:
        kind = normalized.get("kind")

        if kind == "candlestick":
            return self._compute_candlestick(normalized, decision)

        if kind == "orderbook_conventional":
            return self._compute_orderbook(normalized, decision)

        if kind in {"orderbook_large_trades", "orderbook_whale_orders"}:
            return self._compute_event_list(normalized, decision)

        if kind in {"mining_network_health", "onchain_holder_behavior"}:
            return self._compute_metric_timeseries(normalized, decision)

        return {"technical": {}, "statistical": {}, "microstructure": {}, "feature_snapshot": {}}

    def _compute_candlestick(self, normalized: dict[str, Any], decision: dict[str, Any]) -> dict[str, Any]:
        df: pd.DataFrame = normalized["dataframe"]
        result: dict[str, Any] = {
            "technical": {},
            "statistical": {},
            "microstructure": {},
            "feature_snapshot": {},
        }

        feature_frame = pd.DataFrame(index=df.index)

        if decision.get("apply_technical_indicators"):
            technical_df = compute_ohlcv_indicators(df)
            result["technical"] = {
                "last": last_valid_dict(technical_df),
                "columns": list(technical_df.columns),
            }
            feature_frame = pd.concat([feature_frame, technical_df], axis=1)

        if decision.get("apply_statistical_metrics"):
            returns = safe_returns(df["close"], method="log")
            close_stats = rolling_distribution_features(df["close"], self.DEFAULT_WINDOWS)
            return_stats = rolling_distribution_features(returns, self.DEFAULT_WINDOWS)
            return_stats = return_stats.add_prefix("returns_")

            # Rolling beta example: close returns against volume returns.
            # Later this can be BTC vs ETH, NASDAQ, OI, CVD, ETF netflow, etc.
            volume_returns = safe_returns(df["volume"], method="log")
            beta_30 = returns.rolling(30).cov(volume_returns) / volume_returns.rolling(30).var().replace(0, pd.NA)

            statistical_df = pd.concat([close_stats, return_stats], axis=1)
            statistical_df["rolling_beta_return_vs_volume_30"] = beta_30

            result["statistical"] = {
                "close_summary": summarize_series(df["close"]),
                "return_summary": summarize_series(returns),
                "last": last_valid_dict(statistical_df),
                "columns": list(statistical_df.columns),
            }
            feature_frame = pd.concat([feature_frame, statistical_df], axis=1)

        result["feature_snapshot"] = last_valid_dict(feature_frame)
        return result

    def _compute_orderbook(self, normalized: dict[str, Any], decision: dict[str, Any]) -> dict[str, Any]:
        bids: pd.DataFrame = normalized["bids"]
        asks: pd.DataFrame = normalized["asks"]

        micro = orderbook_metrics(bids, asks)
        micro.update(wall_score_from_orderbook(micro))

        # Pure stats over visible liquidity distribution.
        bid_notional_stats = summarize_series(bids["notional_usdt"])
        ask_notional_stats = summarize_series(asks["notional_usdt"])

        feature_snapshot = {**micro}
        feature_snapshot.update({f"bid_notional_{k}": v for k, v in bid_notional_stats.items()})
        feature_snapshot.update({f"ask_notional_{k}": v for k, v in ask_notional_stats.items()})

        return {
            "technical": {},
            "statistical": {
                "bid_notional_summary": bid_notional_stats,
                "ask_notional_summary": ask_notional_stats,
            },
            "microstructure": micro,
            "feature_snapshot": feature_snapshot,
        }

    def _compute_event_list(self, normalized: dict[str, Any], decision: dict[str, Any]) -> dict[str, Any]:
        events: pd.DataFrame = normalized["events"]
        flow = event_flow_metrics(events)

        stats = {}
        for column in ["price", "quantity_btc", "notional_usdt", "age_seconds", "active_duration_seconds"]:
            if column in events.columns:
                stats[column] = summarize_series(events[column])

        feature_snapshot = {**flow}
        for name, summary in stats.items():
            feature_snapshot.update({f"{name}_{k}": v for k, v in summary.items()})

        return {
            "technical": {},
            "statistical": stats,
            "microstructure": flow,
            "feature_snapshot": feature_snapshot,
        }

    def _compute_metric_timeseries(self, normalized: dict[str, Any], decision: dict[str, Any]) -> dict[str, Any]:
        df: pd.DataFrame = normalized["dataframe"]
        numeric_columns = [
            column
            for column in df.columns
            if column not in {"timestamp", "symbol", "timeframe"}
            and pd.api.types.is_numeric_dtype(df[column])
        ]

        feature_frame = pd.DataFrame(index=df.index)
        summaries: dict[str, Any] = {}

        for column in numeric_columns:
            series = pd.to_numeric(df[column], errors="coerce")
            summary = summarize_series(series)
            std = summary.get("std") or 0
            last = summary.get("last")
            mean = summary.get("mean")
            zscore = (last - mean) / std if last is not None and mean is not None and std else 0.0

            summaries[column] = {
                **summary,
                "zscore": float(zscore),
            }

            if len(series.dropna()) >= min(self.DEFAULT_WINDOWS):
                rolling = rolling_distribution_features(series, self.DEFAULT_WINDOWS).add_prefix(f"{column}_")
                feature_frame = pd.concat([feature_frame, rolling], axis=1)

        feature_snapshot: dict[str, Any] = {}
        for column, summary in summaries.items():
            feature_snapshot.update({f"{column}_{key}": value for key, value in summary.items()})
        feature_snapshot.update(last_valid_dict(feature_frame))

        return {
            "technical": {},
            "statistical": {
                "numeric_columns": numeric_columns,
                "summaries": summaries,
                "rolling": {
                    "last": last_valid_dict(feature_frame),
                    "columns": list(feature_frame.columns),
                },
            },
            "microstructure": {},
            "feature_snapshot": feature_snapshot,
        }
