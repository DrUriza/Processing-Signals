from __future__ import annotations
import numpy    as np
import pandas   as pd

try:
    import ripser   as _ripser
    import persim   as _persim
    _RIPSER_AVAILABLE = True
except ImportError:  # optional heavy dependency
    _RIPSER_AVAILABLE = False


def _has_ripser() -> bool:  # noqa: D401
    """Return True when the optional ripser/persim backend is installed."""
    return _RIPSER_AVAILABLE


# ***********************************************************************************************************************
# Functionname:       _validate_hysteresis_thresholds(high_on: float, high_off: float,
#                              low_on: float, low_off: float)
#
# @brief              Validate that hysteresis threshold pairs are logically ordered.
# @pre                high_on and high_off control the upper band; low_on and low_off control the lower band.
# @post               Returns None; raises ValueError if thresholds are inconsistent.
# @param[in]          high_on: Upper trigger threshold
#                     high_off: Upper release threshold
#                     low_on: Lower trigger threshold
#                     low_off: Lower release threshold
# @param[out]         None
#
# @callsequence       @startuml
#                     title _validate_hysteresis_thresholds
#                     start
#                     if (high_on < high_off?) then (yes)
#                       :Raise ValueError;
#                       stop
#                     endif
#                     if (low_off < low_on?) then (yes)
#                       :Raise ValueError;
#                       stop
#                     endif
#                     stop
#                     @enduml
# ***********************************************************************************************************************
def _validate_hysteresis_thresholds(
    high_on: float,
    high_off: float,
    low_on: float,
    low_off: float,
) -> None:
    if high_on < high_off:
        raise ValueError("high_on must be >= high_off")
    if low_off < low_on:
        raise ValueError("low_off must be >= low_on")

# \file ******************************************************************************************************
# COMPANY:            ELATIN
# PROJECT:            SIGNAL_ANALYSIS
# COMPONENT:          MICROSTRUCTURES
# MODULE NAME:        tda_tools.py
# DESCRIPTION:        @brief TDA-inspired utilities for embeddings and Wasserstein-like profiles
# CREATION DATE:      22.04.2026
# VERSION:            $Revision: 0.1$
# CHANGES:            22.04.2026 - Migrated to short AUMOVIO-style function banner comments.
# *************************************************************************************************************

# *************************************************************************************************************
# Functionname:       sliding_window_embedding(signal, dimension: int = 3, delay: int = 1)
#
# @brief              Build a Takens-style sliding-window embedding from a 1D signal.
# @pre                dimension > 0 and delay > 0.
# @post               Returns array with shape (n_windows, dimension) or empty array if not enough samples.
# @param[in]          signal: Input 1D signal
#                     dimension: Embedding dimension
#                     delay: Delay between coordinates
# @param[out]         out: Embedded point cloud as numpy array
#
# @callsequence       @startuml
#                     title sliding_window_embedding
#                     start
#                     :Convert signal to numpy array;
#                     if (dimension <= 0 or delay <= 0?) then (yes)
#                       :Raise ValueError;
#                       stop
#                     endif
#                     :Compute last valid start index;
#                     if (no valid windows?) then (yes)
#                       :Return empty embedding array;
#                       stop
#                     endif
#                     :Build delayed sliding windows;
#                     :Return embedded point cloud;
#                     stop
#                     @enduml
# **************************************************************************************************************
def sliding_window_embedding(signal, dimension: int = 3, delay: int = 1) -> np.ndarray:
    x = np.asarray(signal, dtype=float)
    if dimension <= 0 or delay <= 0:
        raise ValueError("dimension and delay must be > 0")
    # Compute the last valid start index for a full embedded vector.
    last = len(x) - (dimension - 1) * delay
    if last <= 0:
        return np.empty((0, dimension), dtype=float)
    # Build each window with the configured delay stride.
    return np.asarray([x[i : i + dimension * delay : delay] for i in range(last)], dtype=float)

# ***********************************************************************************************************************
# Functionname:       pairwise_distance_matrix(points: np.ndarray)
#
# @brief              Compute dense Euclidean distances between all points.
# @pre                points must be a 2D numpy array.
# @post               Returns square distance matrix (n_points x n_points).
# @param[in]          points: Input point cloud of shape (n_points, n_features)
# @param[out]         out: Pairwise Euclidean distance matrix
#
# @callsequence       @startuml
#                     title pairwise_distance_matrix
#                     start
#                     if (points not 2D?) then (yes)
#                       :Raise ValueError;
#                       stop
#                     endif
#                     if (points empty?) then (yes)
#                       :Return empty matrix;
#                       stop
#                     endif
#                     :Broadcast pairwise differences;
#                     :Compute Euclidean norms;
#                     :Return distance matrix;
#                     stop
#                     @enduml
# ***********************************************************************************************************************
def pairwise_distance_matrix(points: np.ndarray) -> np.ndarray:
    if points.ndim != 2:
        raise ValueError("points must be a 2D array")
    if len(points) == 0:
        return np.empty((0, 0), dtype=float)
    # Use vectorized broadcasting for dense pairwise distances.
    diff = points[:, None, :] - points[None, :, :]
    return np.sqrt(np.sum(diff ** 2, axis=2))

# ***********************************************************************************************************************
# Functionname:       wasserstein_like_distance(a, b, p: int = 1)
#
# @brief              Compute a lightweight p-Wasserstein-like distance on 1D samples.
# @pre                p > 0.
# @post               Returns non-negative scalar distance.
# @param[in]          a: First 1D sample sequence
#                     b: Second 1D sample sequence
#                     p: Wasserstein power
# @param[out]         out: Distance value as float
#
# @callsequence       @startuml
#                     title wasserstein_like_distance
#                     start
#                     :Convert both inputs to flattened numpy arrays;
#                     if (either sample empty?) then (yes)
#                       :Return 0.0;
#                       stop
#                     endif
#                     :Align sample length to common minimum;
#                     if (p <= 0?) then (yes)
#                       :Raise ValueError;
#                       stop
#                     endif
#                     if (p == 1?) then (yes)
#                       :Return mean absolute difference;
#                       stop
#                     endif
#                     :Return p-Wasserstein-like distance;
#                     stop
#                     @enduml
# ***********************************************************************************************************************
def wasserstein_like_distance(a, b, p: int = 1) -> float:
    xa = np.asarray(a, dtype=float).ravel()
    xb = np.asarray(b, dtype=float).ravel()
    if len(xa) == 0 or len(xb) == 0:
        return 0.0
    # Align sample sizes using the common minimum length.
    n = min(len(xa), len(xb))
    xa = np.sort(xa)[:n]
    xb = np.sort(xb)[:n]
    if p <= 0:
        raise ValueError("p must be > 0")
    if p == 1:
        return float(np.mean(np.abs(xa - xb)))
    return float(np.mean(np.abs(xa - xb) ** p) ** (1.0 / p))

# ***********************************************************************************************************************
# Functionname:       rolling_wasserstein_profile(series, window: int = 20, p: int = 1, normalize_to_100: bool = True)
#
# @brief              Build rolling Wasserstein-like profile between adjacent windows.
# @pre                window > 0.
# @post               Returns profile aligned to second window end, forward-filled, remaining NaN set to 0.
# @param[in]          series: Input 1D series
#                     window: Adjacent window length
#                     p: Wasserstein power
#                     normalize_to_100: Normalize profile to [0, 100] when True
# @param[out]         out: Rolling distance profile as pandas Series
#
# @callsequence       @startuml
#                     title rolling_wasserstein_profile
#                     start
#                     :Convert input to pandas Series;
#                     if (window <= 0?) then (yes)
#                       :Raise ValueError;
#                       stop
#                     endif
#                     :Initialize output profile with NaN;
#                     if (not enough samples?) then (yes)
#                       :Return zero-filled profile;
#                       stop
#                     endif
#                     repeat
#                       :Compare adjacent windows with wasserstein_like_distance;
#                       :Store distance and aligned index;
#                     repeat while (more window pairs?)
#                     if (normalize_to_100?) then (yes)
#                       :Normalize values to [0, 100];
#                     endif
#                     :Assign values, forward-fill, and zero-fill output;
#                     :Return profile;
#                     stop
#                     @enduml
# ***********************************************************************************************************************
def rolling_wasserstein_profile(series, window: int = 20, p: int = 1, normalize_to_100: bool = True) -> pd.Series:
    x = pd.Series(series, dtype=float)
    n = len(x)
    if window <= 0:
        raise ValueError("window must be > 0")
    if p <= 0:
        raise ValueError("p must be > 0")
    out = pd.Series(np.nan, index=x.index, name=f"wd_profile_{window}")
    if n < 2 * window:
        return out.fillna(0.0)
    vals = []
    idxs = []
    # Compare adjacent windows [i, i+window) vs [i+window, i+2*window).
    for i in range(0, n - 2 * window + 1):
        a = x.iloc[i : i + window].to_numpy(dtype=float)
        b = x.iloc[i + window : i + 2 * window].to_numpy(dtype=float)
        dist = wasserstein_like_distance(a, b, p=p)
        vals.append(dist)
        idxs.append(x.index[i + 2 * window - 1])
    vals = np.asarray(vals, dtype=float)
    if normalize_to_100 and len(vals) > 0:
        # Normalize robustly with epsilon to avoid zero-division.
        vmin = float(np.min(vals))
        vmax = float(np.max(vals))
        vals = 100.0 * (vals - vmin) / (vmax - vmin + 1e-12)

    out.loc[idxs] = vals
    return out.ffill().fillna(0.0)


# ***********************************************************************************************************************
# Functionname:       wasserstein_regime_signal(distance_profile, high_on: float = 95.0, high_off: float = 80.0,
#                              low_on: float = 20.0, low_off: float = 35.0)
#
# @brief              Convert normalized profile into simple high/low/neutral regime signal.
# @pre                distance_profile should be normalized consistently with configured thresholds.
# @post               Returns signal values in {-1.0, 0.0, 1.0}.
# @param[in]          distance_profile: Wasserstein-like profile values
#                     high_on: Upper trigger threshold
#                     high_off: Upper release threshold
#                     low_on: Lower trigger threshold
#                     low_off: Lower release threshold
# @param[out]         out: Regime signal as pandas Series
#
# @callsequence       @startuml
#                     title wasserstein_regime_signal
#                     start
#                     :Convert distance profile to pandas Series;
#                     :Initialize neutral signal vector;
#                     :Assign high-regime values;
#                     :Assign low-regime values;
#                     :Assign neutral hysteresis band values;
#                     :Return regime signal;
#                     stop
#                     @enduml
# ***********************************************************************************************************************
def wasserstein_regime_signal(
    distance_profile,
    high_on: float = 95.0,
    high_off: float = 80.0,
    low_on: float = 20.0,
    low_off: float = 35.0,
) -> pd.Series:
    _validate_hysteresis_thresholds(
        high_on=high_on,
        high_off=high_off,
        low_on=low_on,
        low_off=low_off,
    )

    wd = pd.Series(distance_profile, dtype=float)
    signal = pd.Series(0.0, index=wd.index, name="wd_regime_signal")

    # Apply simple three-state hysteresis thresholds.
    signal[wd >= high_on] = 1.0
    signal[wd <= low_on] = -1.0
    signal[(wd < high_off) & (wd > low_off)] = 0.0

    return signal


# ***********************************************************************************************************************
# Functionname:       wasserstein_distances(data, window: int = 20, column: str = "close", high_on: float = 95.0,
#                              high_off: float = 80.0, low_on: float = 20.0, low_off: float = 35.0)
#
# @brief              Compute distance profile and regime signal in one call.
# @pre                If data is DataFrame, column must exist.
# @post               Returns tuple(distance_profile, regime_signal).
# @param[in]          data: DataFrame or array-like input source
#                     window: Adjacent window length
#                     column: DataFrame column used when input is DataFrame
#                     use_log_returns: Apply log-returns transform to series before computing profile
#                     high_on: Upper trigger threshold
#                     high_off: Upper release threshold
#                     low_on: Lower trigger threshold
#                     low_off: Lower release threshold
# @param[out]         out: Tuple of (distance_profile, regime_signal)
#
# @callsequence       @startuml
#                     title wasserstein_distances
#                     start
#                     if (data is DataFrame?) then (yes)
#                       if (column missing?) then (yes)
#                         :Raise ValueError;
#                         stop
#                       endif
#                       :Select configured column;
#                     else (no)
#                       :Use raw input as series;
#                     endif
#                     :Call rolling_wasserstein_profile;
#                     :Call wasserstein_regime_signal;
#                     :Return distance profile and regime signal;
#                     stop
#                     @enduml
# ***********************************************************************************************************************
def wasserstein_distances(
    data,
    window: int = 20,
    column: str = "close",
    use_log_returns: bool = False,
    high_on: float = 95.0,
    high_off: float = 80.0,
    low_on: float = 20.0,
    low_off: float = 35.0,
) -> tuple[pd.Series, pd.Series]:
    if isinstance(data, pd.DataFrame):
        if column not in data.columns:
            raise ValueError(f"column '{column}' not found in DataFrame")
        series = data[column]
    else:
        series = data

    series = pd.Series(series, dtype=float)
    if use_log_returns:
        # Log-returns are often more stable than raw prices for regime comparisons.
        series = np.log(series / series.shift(1)).dropna()

    # Fixed defaults are intentionally local to this call.
    wd = rolling_wasserstein_profile(series, window=window, p=1, normalize_to_100=True)
    pos = wasserstein_regime_signal(
        wd,
        high_on=high_on,
        high_off=high_off,
        low_on=low_on,
        low_off=low_off,
    )
    return wd, pos


# ---------------------------------------------------------------------------
# TDA Persistence backend  (ripser/persim when available; lightweight fallback)
# ---------------------------------------------------------------------------

# ***********************************************************************************************************************
# Functionname:       has_ripser()
#
# @brief              Report whether the optional ripser/persim backend is installed.
# @pre                None.
# @post               Returns True when ripser and persim are importable, False otherwise.
# @param[in]          None
# @param[out]         out: Boolean availability flag
#
# @callsequence       @startuml
#                     title has_ripser
#                     start
#                     :Return module-level _RIPSER_AVAILABLE flag;
#                     stop
#                     @enduml
# ***********************************************************************************************************************
def has_ripser() -> bool:
    return _RIPSER_AVAILABLE


# ***********************************************************************************************************************
# Functionname:       compute_persistence_diagram(points: np.ndarray, max_dim: int = 1,
#                              thresh: float | None = None)
#
# @brief              Compute Vietoris-Rips persistence diagrams for a point cloud.
#                     Uses ripser when available; falls back to a distance-matrix approximation.
# @pre                points must be a 2D array with at least 2 rows.
#                     max_dim must be >= 0.
# @post               Returns list of numpy arrays, one per homology dimension,
#                     each with shape (n_bars, 2) containing [birth, death] pairs.
# @param[in]          points: Input point cloud array of shape (n_points, n_features)
#                     max_dim: Maximum homology dimension to compute
#                     thresh: Optional distance threshold for filtration cutoff
# @param[out]         out: List of persistence diagrams per dimension
#
# @callsequence       @startuml
#                     title compute_persistence_diagram
#                     start
#                     :Validate points is 2D with >= 2 rows;
#                     if (ripser available?) then (yes)
#                       :Call ripser.ripser with configured max_dim and thresh;
#                       :Extract dgms list from result;
#                     else (no)
#                       :Compute pairwise distance matrix;
#                       :Build H0 diagram from sorted distances (fallback);
#                     endif
#                     :Return list of persistence diagram arrays;
#                     stop
#                     @enduml
# ***********************************************************************************************************************
def compute_persistence_diagram(
    points: np.ndarray,
    max_dim: int = 1,
    thresh: float | None = None,
) -> list[np.ndarray]:
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or len(pts) < 2:
        raise ValueError("points must be a 2D array with at least 2 rows")
    if max_dim < 0:
        raise ValueError("max_dim must be >= 0")

    if _RIPSER_AVAILABLE:
        kwargs: dict = {"maxdim": max_dim}
        if thresh is not None:
            kwargs["thresh"] = thresh
        result = _ripser.ripser(pts, **kwargs)
        return result["dgms"]

    # --- Lightweight fallback: H0 only via sorted pairwise distances ---
    dist = pairwise_distance_matrix(pts)
    # Upper-triangle distances (excluding diagonal) approximate birth times.
    upper = dist[np.triu_indices(len(pts), k=1)]
    upper_sorted = np.sort(upper)
    # Each pair birth = distance, death = inf (H0 Rips approximation).
    h0 = np.column_stack([upper_sorted, np.full(len(upper_sorted), np.inf)])
    if thresh is not None:
        h0 = h0[h0[:, 0] <= thresh]
    diagrams = [h0] + [np.empty((0, 2), dtype=float)] * max_dim
    return diagrams


# ***********************************************************************************************************************
# Functionname:       persistence_entropy(diagram: np.ndarray)
#
# @brief              Compute Shannon entropy of the persistence lifetimes in a diagram.
#                     Infinite-death bars are excluded from the computation.
# @pre                diagram must be a 2D array of shape (n_bars, 2).
# @post               Returns a non-negative float; returns 0.0 for empty or single-bar diagrams.
# @param[in]          diagram: Persistence diagram array with columns [birth, death]
# @param[out]         out: Persistence entropy as float
#
# @callsequence       @startuml
#                     title persistence_entropy
#                     start
#                     :Convert to numpy array and validate shape;
#                     :Filter out bars with infinite death;
#                     if (no finite bars?) then (yes)
#                       :Return 0.0;
#                       stop
#                     endif
#                     :Compute lifetimes = death - birth;
#                     :Normalize to probability distribution;
#                     :Compute Shannon entropy = -sum(p * log(p));
#                     :Return entropy;
#                     stop
#                     @enduml
# ***********************************************************************************************************************
def persistence_entropy(diagram: np.ndarray) -> float:
    dgm = np.asarray(diagram, dtype=float)
    if dgm.ndim != 2 or dgm.shape[1] != 2:
        raise ValueError("diagram must be a 2D array with shape (n_bars, 2)")
    # Exclude essential (infinite-death) bars.
    finite_mask = np.isfinite(dgm[:, 1])
    dgm_fin = dgm[finite_mask]
    if len(dgm_fin) == 0:
        return 0.0
    lifetimes = dgm_fin[:, 1] - dgm_fin[:, 0]
    lifetimes = lifetimes[lifetimes > 0]
    if len(lifetimes) == 0:
        return 0.0
    total = float(np.sum(lifetimes))
    if total == 0.0:
        return 0.0
    probs = lifetimes / total
    # Clip to avoid log(0).
    probs = np.clip(probs, 1e-15, None)
    return float(-np.sum(probs * np.log(probs)))


# ***********************************************************************************************************************
# Functionname:       rolling_persistence_profile(series, window: int = 20, embed_dim: int = 3,
#                              embed_delay: int = 1, max_dim: int = 1,
#                              normalize_to_100: bool = True)
#
# @brief              Build a rolling persistence-entropy profile over sliding windows.
#                     Each window is embedded using sliding_window_embedding, then a
#                     persistence diagram is computed and summarized by persistence_entropy.
# @pre                window > 0, embed_dim > 0, embed_delay > 0, max_dim >= 0.
# @post               Returns entropy profile aligned to window-end indices,
#                     optionally normalized to [0, 100].
# @param[in]          series: Input 1D series
#                     window: Number of observations per window
#                     embed_dim: Takens embedding dimension
#                     embed_delay: Delay between embedded coordinates
#                     max_dim: Maximum homology dimension for persistence
#                     normalize_to_100: Normalize profile to [0, 100] when True
# @param[out]         out: Rolling persistence entropy profile as pandas Series
#
# @callsequence       @startuml
#                     title rolling_persistence_profile
#                     start
#                     :Validate window, embed_dim, embed_delay, max_dim;
#                     :Convert input to pandas Series;
#                     if (not enough samples?) then (yes)
#                       :Return zero-filled profile;
#                       stop
#                     endif
#                     repeat
#                       :Extract window slice;
#                       :Embed with sliding_window_embedding;
#                       if (embedding has < 2 points?) then (yes)
#                         :Append entropy = 0.0;
#                       else
#                         :Compute persistence diagram;
#                         :Compute persistence_entropy for H1 diagram;
#                         :Append entropy value;
#                       endif
#                     repeat while (more windows?)
#                     if (normalize_to_100?) then (yes)
#                       :Normalize entropy values to [0, 100];
#                     endif
#                     :Build aligned output Series;
#                     :Return rolling persistence profile;
#                     stop
#                     @enduml
# ***********************************************************************************************************************
def rolling_persistence_profile(
    series,
    window: int = 20,
    embed_dim: int = 3,
    embed_delay: int = 1,
    max_dim: int = 1,
    normalize_to_100: bool = True,
) -> pd.Series:
    if window <= 0:
        raise ValueError("window must be > 0")
    if embed_dim <= 0:
        raise ValueError("embed_dim must be > 0")
    if embed_delay <= 0:
        raise ValueError("embed_delay must be > 0")
    if max_dim < 0:
        raise ValueError("max_dim must be >= 0")

    x = pd.Series(series, dtype=float)
    n = len(x)
    out = pd.Series(np.nan, index=x.index, name=f"persist_entropy_{window}")

    if n < window:
        return out.fillna(0.0)

    entropies: list[float] = []
    idxs = []

    for i in range(n - window + 1):
        chunk = x.iloc[i : i + window].to_numpy(dtype=float)
        pts = sliding_window_embedding(chunk, dimension=embed_dim, delay=embed_delay)
        if len(pts) < 2:
            entropies.append(0.0)
        else:
            dgms = compute_persistence_diagram(pts, max_dim=max_dim)
            # Use H1 when available, H0 otherwise.
            dim_idx = min(1, len(dgms) - 1)
            entropies.append(persistence_entropy(dgms[dim_idx]))
        idxs.append(x.index[i + window - 1])

    vals = np.asarray(entropies, dtype=float)
    if normalize_to_100 and len(vals) > 0:
        vmin, vmax = float(np.min(vals)), float(np.max(vals))
        vals = 100.0 * (vals - vmin) / (vmax - vmin + 1e-12)

    out.loc[idxs] = vals
    return out.ffill().fillna(0.0)


__all__ = [
    "sliding_window_embedding",
    "pairwise_distance_matrix",
    "wasserstein_like_distance",
    "rolling_wasserstein_profile",
    "wasserstein_regime_signal",
    "wasserstein_distances",
    # TDA persistence backend
    "compute_persistence_diagram",
    "persistence_entropy",
    "rolling_persistence_profile",
    "has_ripser",
]
