import numpy as np
import pandas as pd

from signal_analysis.transforms.fourier import compute_fft_magnitude
from signal_analysis.transforms.wavelet import wavelet_energy_placeholder
from signal_analysis.microstructures.micro_doppler import micro_doppler_features_placeholder


def test_compute_fft_magnitude_returns_arrays():
    signal = [0.0, 1.0, 0.0, -1.0]
    freqs, mags = compute_fft_magnitude(signal, sample_rate_hz=4.0)
    assert isinstance(freqs, np.ndarray)
    assert isinstance(mags, np.ndarray)
    assert len(freqs) == len(mags)


def test_wavelet_placeholder_returns_series():
    signal = [0.0, 1.0, 0.0, -1.0]
    out = wavelet_energy_placeholder(signal)
    assert isinstance(out, pd.Series)


def test_micro_doppler_placeholder_returns_dataframe():
    signal = [0.0, 1.0, 0.0, -1.0]
    out = micro_doppler_features_placeholder(signal)
    assert isinstance(out, pd.DataFrame)
    assert list(out.columns) == ["length", "energy"]
