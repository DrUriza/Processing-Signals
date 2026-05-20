# signal_analysis

Reusable domain-agnostic signal-analysis toolkit for windowed feature extraction, indicators, transforms, temporal alignment, and compact tabular pipelines.

The library is designed for generic temporal data, including trading, robotics, fluids, radar, and mathematical time series.

## Install (editable)
```bash
pip install -e .
```

## Current Architecture

Core pipeline modules:
- `signal_analysis.core.schema`
- `signal_analysis.core.feature_window`
- `signal_analysis.core.sync`
- `signal_analysis.core.window_builder`
- `signal_analysis.core.feature_pipeline`

Signal feature modules:
- `signal_analysis.indicators.variable_action`
- `signal_analysis.indicators.invariants`
- `signal_analysis.indicators.dynamics`
- `signal_analysis.indicators.states`

## Minimal Example
```python
from __future__ import annotations

import numpy as np
import pandas as pd

from signal_analysis import build_feature_matrix

n = 32
close = np.linspace(100.0, 110.0, n)
open_ = close + 0.1
high = close + 0.5
low = close - 0.5

df = pd.DataFrame(
	{
		"open": open_,
		"high": high,
		"low": low,
		"close": close,
		"signal": np.sin(np.linspace(0.0, 4.0, n)),
	}
)

X = build_feature_matrix(
	df,
	window_size=8,
	step=4,
	value_columns=["close", "signal"],
)

print(X.head())
```

## Notes

- The current compact pipeline focuses on schema validation, canonical feature windows, temporal synchronization, window assembly, and feature-matrix generation.
- Fourier, Wavelet, and TDA modules remain available, but window population for those sections is intentionally not auto-enabled yet.
