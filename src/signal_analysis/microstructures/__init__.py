from signal_analysis.microstructures.micro_doppler import micro_doppler_features_placeholder
from signal_analysis.microstructures.tda_tools import (pairwise_distance_matrix, rolling_wasserstein_profile,
                                                       sliding_window_embedding, wasserstein_distances,
                                                       wasserstein_like_distance, wasserstein_regime_signal,
                                                       compute_persistence_diagram, persistence_entropy,
                                                       rolling_persistence_profile, has_ripser)

# Backward-compatible alias
compute_md_proxy_features = micro_doppler_features_placeholder

__all__ = [
    "micro_doppler_features_placeholder",
    "compute_md_proxy_features",
    "sliding_window_embedding",
    "pairwise_distance_matrix",
    "wasserstein_like_distance",
    "rolling_wasserstein_profile",
    "wasserstein_regime_signal",
    "wasserstein_distances",
    "compute_persistence_diagram",
    "persistence_entropy",
    "rolling_persistence_profile",
    "has_ripser",
]
