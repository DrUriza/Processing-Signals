from __future__ import annotations

from typing import Any

import pandas as pd


def extract_events(
    df: pd.DataFrame,
    numeric_columns: list[str],
    family_key: str,
    data_type: str | None,
    source_name: str | None,
    symbol: str | None,
    timeframe: str | None,
    limit: int = 150,
) -> list[dict[str, Any]]:
    """Extract significant events from z-score outliers in numeric time series."""
    if df.empty:
        return []

    events: list[dict[str, Any]] = []
    for column in [column for column in numeric_columns if column in df.columns]:
        series = pd.to_numeric(df[column], errors="coerce")
        std = series.std()
        if pd.isna(std) or std == 0:
            continue
        zscores = (series - series.mean()) / std
        significant = zscores[zscores.abs() >= 2.0].abs().sort_values(ascending=False).head(25)
        for index in significant.index:
            zscore = zscores.loc[index]
            row = df.loc[index]
            direction = "positive" if zscore >= 0 else "negative"
            events.append(
                {
                    "timestamp": row.get("timestamp"),
                    "symbol": row.get("symbol", symbol),
                    "timeframe": row.get("timeframe", timeframe),
                    "family_key": family_key,
                    "data_type": data_type,
                    "event_type": event_type_for(family_key, column, direction),
                    "metric": column,
                    "value": row.get(column),
                    "severity": "high" if abs(zscore) >= 3.0 else "medium",
                    "direction": direction,
                    "reason": f"zscore {'>=' if zscore >= 0 else '<='} {round(float(zscore), 4)}",
                    "source_name": source_name,
                }
            )
    events.sort(key=lambda event: (str(event.get("timestamp")), event.get("metric", "")))
    return events[:limit]


def event_type_for(family_key: str, metric: str, direction: str) -> str:
    if family_key == "liquidations":
        return "liquidation_spike"
    if family_key == "institutional_flows" and "netflow" in metric:
        return "exchange_netflow_spike"
    if family_key == "mining_network_health":
        return "miner_pressure_event"
    if family_key == "onchain_holder_behavior":
        return "accumulation_distribution_event"
    return "outlier_positive" if direction == "positive" else "outlier_negative"
