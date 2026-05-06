import numpy as np
import pandas as pd
import pytest

from signal_analysis.microstructures import (
    rolling_wasserstein_profile,
    wasserstein_distances,
    wasserstein_regime_signal,
    compute_persistence_diagram,
    persistence_entropy,
    rolling_persistence_profile,
    has_ripser,
)


def test_wasserstein_distances_use_log_returns_preserves_index_alignment():
    df = pd.DataFrame(
        {
            "close": [100, 101, 103, 102, 104, 105, 106, 108, 107, 109],
        }
    )

    wd, pos = wasserstein_distances(df, window=2, use_log_returns=True)

    assert isinstance(wd, pd.Series)
    assert isinstance(pos, pd.Series)
    assert wd.index.equals(pos.index)
    assert set(np.unique(pos.values)).issubset({-1.0, 0.0, 1.0})


def test_rolling_wasserstein_profile_invalid_p_raises_value_error():
    s = pd.Series([1, 2, 3, 4, 5, 6])
    with pytest.raises(ValueError, match="p must be > 0"):
        rolling_wasserstein_profile(s, window=2, p=0)


def test_wasserstein_regime_signal_invalid_hysteresis_raises_value_error():
    wd = pd.Series([10.0, 50.0, 90.0])
    with pytest.raises(ValueError, match="high_on"):
        wasserstein_regime_signal(wd, high_on=70.0, high_off=80.0)

    with pytest.raises(ValueError, match="low_off"):
        wasserstein_regime_signal(wd, low_on=30.0, low_off=20.0)


# ---------------------------------------------------------------------------
# Persistence backend tests (fallback path — no ripser required)
# ---------------------------------------------------------------------------

def test_has_ripser_returns_bool():
    assert isinstance(has_ripser(), bool)


def test_compute_persistence_diagram_fallback_returns_list_of_arrays():
    rng = np.random.default_rng(0)
    pts = rng.random((10, 2))
    dgms = compute_persistence_diagram(pts, max_dim=1)
    assert isinstance(dgms, list)
    assert len(dgms) >= 1
    assert all(isinstance(d, np.ndarray) for d in dgms)
    assert dgms[0].shape[1] == 2


def test_compute_persistence_diagram_invalid_input_raises():
    with pytest.raises(ValueError, match="2D array"):
        compute_persistence_diagram(np.array([1.0, 2.0, 3.0]), max_dim=0)
    with pytest.raises(ValueError, match="2D array"):
        compute_persistence_diagram(np.empty((1, 2)), max_dim=0)


def test_persistence_entropy_finite_bars():
    # birth=0, death=1 bars -> uniform distribution -> entropy = log(n)
    dgm = np.array([[0.0, 1.0], [0.0, 2.0], [0.0, 3.0]])
    ent = persistence_entropy(dgm)
    assert ent > 0.0


def test_persistence_entropy_only_infinite_bars_returns_zero():
    dgm = np.array([[0.0, np.inf], [0.5, np.inf]])
    assert persistence_entropy(dgm) == 0.0


def test_persistence_entropy_empty_diagram_returns_zero():
    dgm = np.empty((0, 2), dtype=float)
    assert persistence_entropy(dgm) == 0.0


def test_rolling_persistence_profile_shape_and_range():
    rng = np.random.default_rng(42)
    s = pd.Series(rng.standard_normal(50))
    profile = rolling_persistence_profile(s, window=10, embed_dim=3, embed_delay=1, normalize_to_100=True)
    assert isinstance(profile, pd.Series)
    assert len(profile) == len(s)
    assert profile.index.equals(s.index)
    assert profile.min() >= 0.0
    assert profile.max() <= 100.0 + 1e-9


def test_rolling_persistence_profile_too_short_returns_zeros():
    s = pd.Series([1.0, 2.0, 3.0])
    profile = rolling_persistence_profile(s, window=20)
    assert (profile == 0.0).all()


def test_rolling_persistence_profile_invalid_params_raise():
    s = pd.Series(range(30))
    with pytest.raises(ValueError, match="window"):
        rolling_persistence_profile(s, window=0)
    with pytest.raises(ValueError, match="embed_dim"):
        rolling_persistence_profile(s, embed_dim=0)
    with pytest.raises(ValueError, match="embed_delay"):
        rolling_persistence_profile(s, embed_delay=0)
    with pytest.raises(ValueError, match="max_dim"):
        rolling_persistence_profile(s, max_dim=-1)
