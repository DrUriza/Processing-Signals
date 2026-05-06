from __future__ import annotations

import math
import numpy as np
import pandas as pd

from signal_analysis.utils.helpers import to_series, validate_window


# \file **********************************************************************
# COMPANY:            ELATIN
# PROJECT:            SIGNAL_ANALYSIS
# COMPONENT:          TRANSFORMS
# MODULE NAME:        wavelet.py
# DESCRIPTION:        @brief Haar-based wavelet transform utilities
# CREATION DATE:      22.04.2026
# VERSION:            $Revision: 0.2$
# CHANGES:            22.04.2026 - Migrated to short banner comments.
#                     22.04.2026 - Refactored to class-based OOP.
# *****************************************************************************


class WaveletTransforms:

    # ***********************************************************************************************************************
    # Functionname:       WaveletTransforms.moving_window_energy(series, window_size: int = 4,
    #                              min_periods: int | None = None, normalize: bool = False)
    #
    # @brief              Compute rolling sum of squared amplitudes.
    # @pre                window_size > 0.
    # @post               Returns rolling energy series named window_energy_window_size.
    # @param[in]          series: Input signal
    #                     window_size: Rolling window length
    #                     min_periods: Minimum periods for rolling window
    #                     normalize: Divide by window_size when True
    # @param[out]         out: Rolling energy as pandas Series
    #
    # @callsequence       @startuml
    #                     title WaveletTransforms.moving_window_energy
    #                     start
    #                     :Validate window_size;
    #                     :Convert input to pandas Series;
    #                     :Resolve rolling min_periods;
    #                     :Compute rolling squared-energy sum;
    #                     if (normalize?) then (yes)
    #                       :Divide energy by window_size;
    #                     endif
    #                     :Rename and return energy series;
    #                     stop
    #                     @enduml
    # ***********************************************************************************************************************
    @staticmethod
    def moving_window_energy(
        series,
        window_size: int = 4,
        min_periods: int | None = None,
        normalize: bool = False,
    ) -> pd.Series:
        validate_window(window_size, "window_size")
        s = to_series(series, name="signal")
        mp = window_size if min_periods is None else min_periods

        # Sum of squared samples in each rolling window.
        energy = (s ** 2).rolling(window=window_size, min_periods=mp).sum()
        if normalize:
            energy = energy / float(window_size)

        energy.name = f"window_energy_{window_size}"
        return energy

    # ***********************************************************************************************************************
    # Functionname:       WaveletTransforms._haar_step(values: np.ndarray)
    #
    # @brief              Execute one-level Haar decomposition step.
    # @pre                values is a 1D numpy array.
    # @post               Returns approximation and detail coefficients; drops last sample if odd length.
    # @param[in]          values: Input samples
    # @param[out]         out: Tuple(approximation, detail)
    #
    # @callsequence       @startuml
    #                     title WaveletTransforms._haar_step
    #                     start
    #                     if (len(values) < 2?) then (yes)
    #                       :Return empty approximation/detail arrays;
    #                       stop
    #                     endif
    #                     if (odd length?) then (yes)
    #                       :Drop last sample;
    #                     endif
    #                     :Compute Haar approximation coefficients;
    #                     :Compute Haar detail coefficients;
    #                     :Return approximation and detail arrays;
    #                     stop
    #                     @enduml
    # ***********************************************************************************************************************
    @staticmethod
    def _haar_step(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n = len(values)
        if n < 2:
            return np.array([], dtype=float), np.array([], dtype=float)

        if n % 2 != 0:
            values = values[:-1]

        # Haar scaling: normalize by sqrt(2) for energy preservation.
        approx = (values[0::2] + values[1::2]) / math.sqrt(2.0)
        detail = (values[0::2] - values[1::2]) / math.sqrt(2.0)
        return approx, detail

    # ***********************************************************************************************************************
    # Functionname:       WaveletTransforms.compute_haar_wavelet_decomposition(series, levels: int = 1)
    #
    # @brief              Compute multi-level Haar decomposition.
    # @pre                levels > 0.
    # @post               Returns dict with approximation and detail_level_i series.
    # @param[in]          series: Input signal
    #                     levels: Number of decomposition levels
    # @param[out]         out: Dictionary of decomposition coefficients
    #
    # @callsequence       @startuml
    #                     title WaveletTransforms.compute_haar_wavelet_decomposition
    #                     start
    #                     :Validate levels;
    #                     :Convert input to pandas Series and numpy array;
    #                     repeat
    #                       :Call _haar_step;
    #                       :Store detail coefficients for current level;
    #                       :Promote approximation for next iteration;
    #                     repeat while (more levels and enough samples?)
    #                     :Build result dictionary with approximation and details;
    #                     :Return decomposition dictionary;
    #                     stop
    #                     @enduml
    # ***********************************************************************************************************************
    @staticmethod
    def compute_haar_wavelet_decomposition(
        series,
        levels: int = 1,
    ) -> dict[str, pd.Series]:
        validate_window(levels, "levels")
        s = to_series(series, name="signal")
        current = s.to_numpy(dtype=float)

        details: dict[str, pd.Series] = {}

        # Iteratively decompose and collect detail coefficients per level.
        for level in range(1, levels + 1):
            approx, detail = WaveletTransforms._haar_step(current)
            details[f"detail_level_{level}"] = pd.Series(
                detail,
                name=f"detail_level_{level}",
                dtype=float,
            )
            current = approx
            if len(current) < 2:
                break

        result: dict[str, pd.Series] = {
            "approximation": pd.Series(current, name="approximation", dtype=float)
        }
        result.update(details)
        return result

    # ***********************************************************************************************************************
    # Functionname:       WaveletTransforms.compute_wavelet_energy(series, levels: int = 1,
    #                              normalize: bool = False)
    #
    # @brief              Compute energy summary from Haar decomposition coefficients.
    # @pre                levels > 0.
    # @post               Returns approximation/detail energy series, optionally normalized.
    # @param[in]          series: Input signal
    #                     levels: Number of decomposition levels
    #                     normalize: Normalize energies by total energy when True
    # @param[out]         out: Wavelet energy summary as pandas Series
    #
    # @callsequence       @startuml
    #                     title WaveletTransforms.compute_wavelet_energy
    #                     start
    #                     :Call compute_haar_wavelet_decomposition;
    #                     :Compute approximation energy;
    #                     :Accumulate detail energies by level;
    #                     if (normalize?) then (yes)
    #                       :Normalize energies by total energy;
    #                     endif
    #                     :Return wavelet energy summary;
    #                     stop
    #                     @enduml
    # ***********************************************************************************************************************
    @staticmethod
    def compute_wavelet_energy(
        series,
        levels: int = 1,
        normalize: bool = False,
    ) -> pd.Series:
        validate_window(levels, "levels")
        coeffs = WaveletTransforms.compute_haar_wavelet_decomposition(series, levels=levels)

        energies: dict[str, float] = {}
        total_energy = 0.0

        approx = coeffs["approximation"].dropna().to_numpy(dtype=float)
        approx_energy = float(np.sum(approx ** 2))
        energies["approximation_energy"] = approx_energy
        total_energy += approx_energy

        # Accumulate energy from each detail level.
        detail_keys = sorted(k for k in coeffs.keys() if k.startswith("detail_level_"))
        for key in detail_keys:
            values = coeffs[key].dropna().to_numpy(dtype=float)
            energy = float(np.sum(values ** 2))
            energies[f"{key}_energy"] = energy
            total_energy += energy

        if normalize and total_energy > 0:
            energies = {k: v / total_energy for k, v in energies.items()}

        return pd.Series(energies, name="wavelet_energy")

    # ***********************************************************************************************************************
    # Functionname:       WaveletTransforms.wavelet_energy_placeholder(series, levels: int = 1)
    #
    # @brief              Backward-compatible wrapper around compute_wavelet_energy.
    # @pre                levels > 0.
    # @post               Returns non-normalized wavelet energy summary.
    # @param[in]          series: Input signal
    #                     levels: Number of decomposition levels
    # @param[out]         out: Wavelet energy summary as pandas Series
    #
    # @callsequence       @startuml
    #                     title WaveletTransforms.wavelet_energy_placeholder
    #                     start
    #                     :Call compute_wavelet_energy with normalize=False;
    #                     :Return wavelet energy summary;
    #                     stop
    #                     @enduml
    # ***********************************************************************************************************************
    @staticmethod
    def wavelet_energy_placeholder(
        series,
        levels: int = 1,
    ) -> pd.Series:
        return WaveletTransforms.compute_wavelet_energy(series, levels=levels, normalize=False)


# ---------------------------------------------------------------------------
# Backward-compatible module-level wrappers
# ---------------------------------------------------------------------------
def moving_window_energy(series, window_size: int = 4, min_periods: int | None = None, normalize: bool = False) -> pd.Series:
    return WaveletTransforms.moving_window_energy(series, window_size=window_size, min_periods=min_periods, normalize=normalize)


def _haar_step(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return WaveletTransforms._haar_step(values)


def compute_haar_wavelet_decomposition(series, levels: int = 1) -> dict[str, pd.Series]:
    return WaveletTransforms.compute_haar_wavelet_decomposition(series, levels=levels)


def compute_wavelet_energy(series, levels: int = 1, normalize: bool = False) -> pd.Series:
    return WaveletTransforms.compute_wavelet_energy(series, levels=levels, normalize=normalize)


def wavelet_energy_placeholder(series, levels: int = 1) -> pd.Series:
    return WaveletTransforms.wavelet_energy_placeholder(series, levels=levels)


__all__ = [
    "WaveletTransforms",
    "moving_window_energy",
    "compute_haar_wavelet_decomposition",
    "compute_wavelet_energy",
    "wavelet_energy_placeholder",
]
