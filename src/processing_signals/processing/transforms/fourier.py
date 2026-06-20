from __future__                    import annotations
import numpy                       as np
import pandas                      as pd
from signal_analysis.utils.helpers import to_series


# \file **********************************************************************
# COMPANY:            ELATIN
# PROJECT:            SIGNAL_ANALYSIS
# COMPONENT:          TRANSFORMS
# MODULE NAME:        fourier.py
# DESCRIPTION:        @brief FFT-based transform and spectral features
# CREATION DATE:      22.04.2026
# VERSION:            $Revision: 0.2$
# CHANGES:            22.04.2026 - Migrated to short banner comments.
#                     22.04.2026 - Refactored to class-based OOP.
# *****************************************************************************


class FourierTransforms:

    # ***********************************************************************************************************************
    # Functionname:       FourierTransforms.compute_fft_magnitude(series, sample_rate_hz: float = 1.0,
    #                              detrend: bool = False, normalize: bool = True)
    #
    # @brief              Compute one-sided FFT magnitude spectrum.
    # @pre                sample_rate_hz > 0.
    # @post               Returns tuple(freqs, magnitudes).
    # @param[in]          series: Input signal
    #                     sample_rate_hz: Sampling rate in Hz
    #                     detrend: Remove mean before FFT when True
    #                     normalize: Divide magnitude by signal length when True
    # @param[out]         out: Tuple of frequency and magnitude arrays
    #
    # @callsequence       @startuml
    #                     title FourierTransforms.compute_fft_magnitude
    #                     start
    #                     if (sample_rate_hz <= 0?) then (yes)
    #                       :Raise ValueError;
    #                       stop
    #                     endif
    #                     :Convert input to numpy signal array;
    #                     if (signal empty?) then (yes)
    #                       :Return empty frequency and magnitude arrays;
    #                       stop
    #                     endif
    #                     if (detrend?) then (yes)
    #                       :Subtract signal mean;
    #                     endif
    #                     :Compute rFFT, frequencies, and magnitudes;
    #                     if (normalize?) then (yes)
    #                       :Divide magnitudes by signal length;
    #                     endif
    #                     :Return frequency and magnitude arrays;
    #                     stop
    #                     @enduml
    # ***********************************************************************************************************************
    @staticmethod
    def compute_fft_magnitude(series, sample_rate_hz: float = 1.0, detrend: bool = False, normalize: bool = True) -> tuple[np.ndarray, np.ndarray]:
        if sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be > 0")

        s = to_series(series, name="signal").to_numpy(dtype=float)
        if len(s) == 0:
            return np.array([], dtype=float), np.array([], dtype=float)

        if detrend:
            s = s - np.mean(s)

        fft_vals = np.fft.rfft(s)
        freqs = np.fft.rfftfreq(len(s), d=1.0 / sample_rate_hz)
        mags = np.abs(fft_vals)

        if normalize:
            mags = mags / len(s)

        return freqs, mags

    # ***********************************************************************************************************************
    # Functionname:       FourierTransforms.compute_fft_power_spectrum(series, sample_rate_hz: float = 1.0,
    #                              detrend: bool = False, normalize: bool = True)
    #
    # @brief              Compute one-sided FFT power spectrum.
    # @pre                Same preconditions as compute_fft_magnitude.
    # @post               Returns tuple(freqs, power).
    # @param[in]          series: Input signal
    #                     sample_rate_hz: Sampling rate in Hz
    #                     detrend: Remove mean before FFT when True
    #                     normalize: Divide magnitude by signal length when True
    # @param[out]         out: Tuple of frequency and power arrays
    #
    # @callsequence       @startuml
    #                     title FourierTransforms.compute_fft_power_spectrum
    #                     start
    #                       :Call compute_fft_magnitude;
    #                       :Square magnitudes to get power;
    #                       :Return frequency and power arrays;
    #                     stop
    #                     @enduml
    # ***********************************************************************************************************************
    @staticmethod
    def compute_fft_power_spectrum(
        series,
        sample_rate_hz: float = 1.0,
        detrend: bool = False,
        normalize: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        # Power is squared magnitude.
        freqs, mags = FourierTransforms.compute_fft_magnitude(
            series,
            sample_rate_hz=sample_rate_hz,
            detrend=detrend,
            normalize=normalize,
        )
        power = mags ** 2
        return freqs, power

    # ***********************************************************************************************************************
    # Functionname:       FourierTransforms.dominant_frequency(series, sample_rate_hz: float = 1.0,
    #                              detrend: bool = False, ignore_dc: bool = True)
    #
    # @brief              Estimate dominant frequency from magnitude spectrum.
    # @pre                sample_rate_hz > 0.
    # @post               Returns dominant frequency in Hz, or 0.0 for empty input.
    # @param[in]          series: Input signal
    #                     sample_rate_hz: Sampling rate in Hz
    #                     detrend: Remove mean before FFT when True
    #                     ignore_dc: Ignore zero-frequency bin when True
    # @param[out]         out: Dominant frequency in Hz
    #
    # @callsequence       @startuml
    #                     title FourierTransforms.dominant_frequency
    #                     start
    #                     :Call compute_fft_magnitude;
    #                     if (frequency array empty?) then (yes)
    #                       :Return 0.0;
    #                       stop
    #                     endif
    #                     if (ignore_dc?) then (yes)
    #                       :Skip first FFT bin when possible;
    #                     endif
    #                     :Locate index of maximum magnitude;
    #                     :Return dominant frequency;
    #                     stop
    #                     @enduml
    # ***********************************************************************************************************************
    @staticmethod
    def dominant_frequency(series, sample_rate_hz: float = 1.0, detrend: bool = False, ignore_dc: bool = True) -> float:
        freqs, mags = FourierTransforms.compute_fft_magnitude(
            series,
            sample_rate_hz=sample_rate_hz,
            detrend=detrend,
            normalize=True)

        if len(freqs) == 0:
            return 0.0

        # Optionally skip DC component at index 0.
        start = 1 if ignore_dc and len(freqs) > 1 else 0
        idx = int(np.argmax(mags[start:])) + start
        return float(freqs[idx])

    # ***********************************************************************************************************************
    # Functionname:       FourierTransforms.spectral_energy(series, sample_rate_hz: float = 1.0,
    #                              detrend: bool = False)
    #
    # @brief              Compute total spectral energy from one-sided FFT power.
    # @pre                sample_rate_hz > 0.
    # @post               Returns non-negative scalar energy.
    # @param[in]          series: Input signal
    #                     sample_rate_hz: Sampling rate in Hz
    #                     detrend: Remove mean before FFT when True
    # @param[out]         out: Total spectral energy
    #
    # @callsequence       @startuml
    #                     title FourierTransforms.spectral_energy
    #                     start
    #                     :Call compute_fft_power_spectrum;
    #                     :Sum power values;
    #                     :Return spectral energy;
    #                     stop
    #                     @enduml
    # ***********************************************************************************************************************
    @staticmethod
    def spectral_energy(series, sample_rate_hz: float = 1.0, detrend: bool = False) -> float:
        _, power = FourierTransforms.compute_fft_power_spectrum(series, sample_rate_hz=sample_rate_hz, detrend=detrend, normalize=True)
        return float(np.sum(power))

    # ***********************************************************************************************************************
    # Functionname:       FourierTransforms.spectral_centroid(series, sample_rate_hz: float = 1.0,
    #                              detrend: bool = False)
    #
    # @brief              Compute spectral centroid in Hz.
    # @pre                sample_rate_hz > 0.
    # @post               Returns 0.0 for empty or zero-magnitude spectra.
    # @param[in]          series: Input signal
    #                     sample_rate_hz: Sampling rate in Hz
    #                     detrend: Remove mean before FFT when True
    # @param[out]         out: Spectral centroid in Hz
    #
    # @callsequence       @startuml
    #                     title FourierTransforms.spectral_centroid
    #                     start
    #                     :Call compute_fft_magnitude;
    #                     :Compute magnitude sum;
    #                     if (empty spectrum or zero denominator?) then (yes)
    #                       :Return 0.0;
    #                       stop
    #                     endif
    #                     :Compute weighted frequency average;
    #                     :Return spectral centroid;
    #                     stop
    #                     @enduml
    # ***********************************************************************************************************************
    @staticmethod
    def spectral_centroid(series, sample_rate_hz: float = 1.0, detrend: bool = False) -> float:
        freqs, mags = FourierTransforms.compute_fft_magnitude(series, sample_rate_hz=sample_rate_hz, detrend=detrend, normalize=True)
        denom = float(np.sum(mags))
        if len(freqs) == 0 or denom == 0.0:
            return 0.0
        # Weighted average frequency by magnitude.
        return float(np.sum(freqs * mags) / denom)

    # ***********************************************************************************************************************
    # Functionname:       FourierTransforms.fft_feature_summary(series, sample_rate_hz: float = 1.0,
    #                              detrend: bool = False)
    #
    # @brief              Compute compact FFT feature summary.
    # @pre                sample_rate_hz > 0.
    # @post               Returns summary with dominant_frequency, spectral_energy, spectral_centroid.
    # @param[in]          series: Input signal
    #                     sample_rate_hz: Sampling rate in Hz
    #                     detrend: Remove mean before FFT when True
    # @param[out]         out: Feature summary as pandas Series
    #
    # @callsequence       @startuml
    #                     title FourierTransforms.fft_feature_summary
    #                     start
    #                     :Call dominant_frequency;
    #                     :Call spectral_energy;
    #                     :Call spectral_centroid;
    #                     :Build summary Series;
    #                     :Return FFT summary;
    #                     stop
    #                     @enduml
    # ***********************************************************************************************************************
    @staticmethod
    def fft_feature_summary(series, sample_rate_hz: float = 1.0, detrend: bool = False) -> pd.Series:
        return pd.Series(
            {
                "dominant_frequency": FourierTransforms.dominant_frequency(
                    series,
                    sample_rate_hz=sample_rate_hz,
                    detrend=detrend,
                ),
                "spectral_energy": FourierTransforms.spectral_energy(
                    series,
                    sample_rate_hz=sample_rate_hz,
                    detrend=detrend,
                ),
                "spectral_centroid": FourierTransforms.spectral_centroid(
                    series,
                    sample_rate_hz=sample_rate_hz,
                    detrend=detrend,
                ),
            },
            name="fft_summary",
        )


# ---------------------------------------------------------------------------
# Backward-compatible module-level wrappers
# ---------------------------------------------------------------------------
def compute_fft_magnitude(series, sample_rate_hz: float = 1.0, detrend: bool = False, normalize: bool = True) -> tuple[np.ndarray, np.ndarray]:
    return FourierTransforms.compute_fft_magnitude(series, sample_rate_hz=sample_rate_hz, detrend=detrend, normalize=normalize)


def compute_fft_power_spectrum(series, sample_rate_hz: float = 1.0, detrend: bool = False, normalize: bool = True) -> tuple[np.ndarray, np.ndarray]:
    return FourierTransforms.compute_fft_power_spectrum(series, sample_rate_hz=sample_rate_hz, detrend=detrend, normalize=normalize)


def dominant_frequency(series, sample_rate_hz: float = 1.0, detrend: bool = False, ignore_dc: bool = True) -> float:
    return FourierTransforms.dominant_frequency(series, sample_rate_hz=sample_rate_hz, detrend=detrend, ignore_dc=ignore_dc)


def spectral_energy(series, sample_rate_hz: float = 1.0, detrend: bool = False) -> float:
    return FourierTransforms.spectral_energy(series, sample_rate_hz=sample_rate_hz, detrend=detrend)


def spectral_centroid(series, sample_rate_hz: float = 1.0, detrend: bool = False) -> float:
    return FourierTransforms.spectral_centroid(series, sample_rate_hz=sample_rate_hz, detrend=detrend)


def fft_feature_summary(series, sample_rate_hz: float = 1.0, detrend: bool = False) -> pd.Series:
    return FourierTransforms.fft_feature_summary(series, sample_rate_hz=sample_rate_hz, detrend=detrend)


__all__ = [
    "FourierTransforms",
    "compute_fft_magnitude",
    "compute_fft_power_spectrum",
    "dominant_frequency",
    "spectral_energy",
    "spectral_centroid",
    "fft_feature_summary",
]
