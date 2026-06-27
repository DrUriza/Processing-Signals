from __future__ import annotations

from typing import Any

import pandas as pd

from processing_signals.processing.math.statistics import (
    EXCLUDED_COLUMNS,
    compute_block_statistics,
    last_valid_dict,
    summarize_series,
)
from processing_signals.processing.math.statistical_regimes import compute_statistical_regimes
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

    DEFAULT_WINDOWS = [20, 50, 100]

    def compute(self, normalized: dict[str, Any], decision: dict[str, Any]) -> dict[str, Any]:
        kind = normalized.get("kind")
        result = self._base_result()

        if kind == "candlestick":
            result.update(self._compute_candlestick(normalized, decision))
        elif kind == "orderbook_conventional":
            result.update(self._compute_orderbook(normalized, decision))
        elif kind in {"orderbook_large_trades", "orderbook_whale_orders"}:
            result.update(self._compute_event_list(normalized, decision))

        frame = self._frame_for_statistics(normalized)
        if not frame.empty:
            statistics = compute_block_statistics(
                frame,
                windows=self.DEFAULT_WINDOWS,
                excluded_columns=EXCLUDED_COLUMNS,
            )
            regimes = compute_statistical_regimes(
                frame,
                windows=self.DEFAULT_WINDOWS,
                excluded_columns=EXCLUDED_COLUMNS,
            )
            result["statistics"] = statistics
            result["statistical_regimes"] = regimes
            # Keep the legacy key for compatibility while introducing math.statistics.
            result["statistical"] = statistics
            result["feature_snapshot"].update(statistics.get("last", {}))

        return result

    @staticmethod
    def _base_result() -> dict[str, Any]:
        return {
            "technical": {},
            "statistical": {},
            "statistics": {},
            "statistical_regimes": {},
            "microstructure": {},
            "feature_snapshot": {},
        }

    def _compute_candlestick(self, normalized: dict[str, Any], decision: dict[str, Any]) -> dict[str, Any]:
        df: pd.DataFrame = normalized["dataframe"]
        result: dict[str, Any] = self._base_result()

        feature_frame = pd.DataFrame(index=df.index)

        if decision.get("apply_technical_indicators"):
            technical_df = compute_ohlcv_indicators(df)
            result["technical"] = {
                "last": last_valid_dict(technical_df),
                "columns": list(technical_df.columns),
            }
            feature_frame = pd.concat([feature_frame, technical_df], axis=1)

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
            "statistics": {},
            "statistical_regimes": {},
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
            "statistics": {},
            "statistical_regimes": {},
            "microstructure": flow,
            "feature_snapshot": feature_snapshot,
        }

    def _frame_for_statistics(self, normalized: dict[str, Any]) -> pd.DataFrame:
        if isinstance(normalized.get("dataframe"), pd.DataFrame):
            return normalized["dataframe"].copy()

        if isinstance(normalized.get("events"), pd.DataFrame):
            return normalized["events"].copy()

        bids = normalized.get("bids")
        asks = normalized.get("asks")
        if isinstance(bids, pd.DataFrame) and isinstance(asks, pd.DataFrame):
            bid_df = bids.add_prefix("bid_").reset_index(drop=True)
            ask_df = asks.add_prefix("ask_").reset_index(drop=True)
            return pd.concat([bid_df, ask_df], axis=1)

        return pd.DataFrame()
