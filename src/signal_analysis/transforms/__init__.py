from signal_analysis.transforms.fourier import (FourierTransforms, compute_fft_magnitude, compute_fft_power_spectrum,
                                                dominant_frequency, fft_feature_summary, spectral_centroid, spectral_energy)
from signal_analysis.transforms.wavelet import (WaveletTransforms, compute_haar_wavelet_decomposition, compute_wavelet_energy,
                                                moving_window_energy, wavelet_energy_placeholder)

__all__ = [
    # Classes
    "FourierTransforms",
    "WaveletTransforms",
    # Backward-compat functions
    "compute_fft_magnitude",
    "compute_fft_power_spectrum",
    "dominant_frequency",
    "spectral_energy",
    "spectral_centroid",
    "fft_feature_summary",
    "moving_window_energy",
    "compute_haar_wavelet_decomposition",
    "compute_wavelet_energy",
    "wavelet_energy_placeholder",
]
