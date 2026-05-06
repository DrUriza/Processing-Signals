from __future__ import annotations

import numpy as np
import pandas as pd

from signal_analysis.utils.helpers import to_series


# \file **********************************************************************
# COMPANY:            ELATIN
# PROJECT:            SIGNAL_ANALYSIS
# COMPONENT:          FILTERS
# MODULE NAME:        kalman.py
# DESCRIPTION:        @brief Causal Kalman filtering utilities
# CREATION DATE:      22.04.2026
# VERSION:            $Revision: 0.1$
# CHANGES:            22.04.2026 - Migrated to short banner comments.
# *****************************************************************************


# ***********************************************************************************************************************
# Functionname:       _validate_positive_float(value: float, name: str)
#
# @brief              Validate that scalar configuration values are strictly positive.
# @pre                None
# @post               Raises ValueError when value <= 0
# @param[in]          value: Numeric value to validate
#                     name: Parameter name used in the error message
# @param[out]         result: None
# ***********************************************************************************************************************
def _validate_positive_float(value: float, name: str) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive.")


class KalmanFilters:
    # *******************************************************************************************************************
    # Functionname:       KalmanFilters.filter_1d(series, process_var: float = 1e-5,
    #                              measurement_var: float = 1e-1, initial_cov: float = 1e-1,
    #                              dt: float = 1.0, return_velocity: bool = False)
    #
    # @brief              Filter a 1D series with a constant-velocity Kalman model.
    # @pre                process_var > 0, measurement_var > 0, initial_cov > 0, dt > 0
    # @post               Returns estimate series or estimate/velocity DataFrame
    # @param[in]          series: Input scalar sequence
    #                     process_var: Process noise variance
    #                     measurement_var: Measurement noise variance
    #                     initial_cov: Initial covariance scale
    #                     dt: Time step
    #                     return_velocity: Include velocity output when True
    # @param[out]         result: Filtered output as Series or DataFrame
    #
    # @callsequence       @startuml
    #                     title KalmanFilters.filter_1d
    #                     start
    #                     :Validate scalar configuration values;
    #                     :Convert input to pd.Series and numpy array;
    #                     :Initialize state-space matrices and covariances;
    #                     repeat
    #                       :Predict state and covariance;
    #                       :Update with current measurement;
    #                       :Store estimate and velocity;
    #                     repeat while (samples remaining)
    #                     if (return_velocity?) then (yes)
    #                       :Return DataFrame(estimate, velocity);
    #                     else (no)
    #                       :Return estimate Series;
    #                     endif
    #                     end
    #                     @enduml
    # *******************************************************************************************************************
    @staticmethod
    def filter_1d(
        series,
        process_var: float = 1e-5,
        measurement_var: float = 1e-1,
        initial_cov: float = 1e-1,
        dt: float = 1.0,
        return_velocity: bool = False,
    ) -> pd.Series | pd.DataFrame:
        _validate_positive_float(process_var, "process_var")
        _validate_positive_float(measurement_var, "measurement_var")
        _validate_positive_float(initial_cov, "initial_cov")
        _validate_positive_float(dt, "dt")

        s = to_series(series, name="signal")
        if s.empty:
            raise ValueError("Input series must not be empty.")

        idx = s.index
        z_values = s.to_numpy(dtype=float)

        F = np.array([[1.0, dt], [0.0, 1.0]], dtype=float)
        H = np.array([[1.0, 0.0]], dtype=float)
        Q = process_var * np.eye(2, dtype=float)
        R = np.array([[measurement_var]], dtype=float)

        x = np.array([z_values[0], 0.0], dtype=float)
        P = initial_cov * np.eye(2, dtype=float)

        estimate = np.full(len(s), np.nan, dtype=float)
        velocity = np.full(len(s), np.nan, dtype=float)

        for i, z in enumerate(z_values):
            x_pred = F @ x
            P_pred = F @ P @ F.T + Q

            y = np.array([z], dtype=float) - H @ x_pred
            S = H @ P_pred @ H.T + R
            K = P_pred @ H.T @ np.linalg.inv(S)

            x = x_pred + K @ y
            P = (np.eye(2, dtype=float) - K @ H) @ P_pred

            estimate[i] = x[0]
            velocity[i] = x[1]

        if return_velocity:
            return pd.DataFrame(
                {
                    "estimate": estimate,
                    "velocity": velocity,
                },
                index=idx,
            )

        return pd.Series(estimate, index=idx, name="kalman_1d")

    # *******************************************************************************************************************
    # Functionname:       KalmanFilters.filter_multivariate(data: pd.DataFrame, columns: list[str] | None = None,
    #                              process_var: float = 1e-5, measurement_var: float = 1e-1,
    #                              initial_cov: float = 1e-1, dt: float = 1.0,
    #                              return_velocity: bool = False)
    #
    # @brief              Filter multiple observed columns jointly in a multivariate Kalman model.
    # @pre                Positive variances/time-step and at least one numeric input column
    # @post               Returns filtered columns and optional velocity columns
    # @param[in]          data: Input DataFrame
    #                     columns: Columns to filter, all numeric columns when None
    #                     process_var: Process noise variance
    #                     measurement_var: Measurement noise variance
    #                     initial_cov: Initial covariance scale
    #                     dt: Time step
    #                     return_velocity: Include velocity outputs when True
    # @param[out]         result: Filtered DataFrame
    #
    # @callsequence       @startuml
    #                     title KalmanFilters.filter_multivariate
    #                     start
    #                     :Validate scalar configuration values;
    #                     :Resolve/validate numeric columns;
    #                     :Build multivariate state-space matrices;
    #                     repeat
    #                       :Predict state and covariance;
    #                       :Update with vector measurement;
    #                       :Store estimates and velocities;
    #                     repeat while (rows remaining)
    #                     :Build output DataFrame;
    #                     if (return_velocity?) then (yes)
    #                       :Append velocity columns;
    #                     endif
    #                     :Return filtered DataFrame;
    #                     end
    #                     @enduml
    # *******************************************************************************************************************
    @staticmethod
    def filter_multivariate(
        data: pd.DataFrame,
        columns: list[str] | None = None,
        process_var: float = 1e-5,
        measurement_var: float = 1e-1,
        initial_cov: float = 1e-1,
        dt: float = 1.0,
        return_velocity: bool = False,
    ) -> pd.DataFrame:
        _validate_positive_float(process_var, "process_var")
        _validate_positive_float(measurement_var, "measurement_var")
        _validate_positive_float(initial_cov, "initial_cov")
        _validate_positive_float(dt, "dt")

        if columns is None:
            columns = data.select_dtypes(include=[np.number]).columns.tolist()

        if not columns:
            raise ValueError("No numeric columns available to filter.")

        obs = data[columns].astype(float).copy()
        if obs.empty:
            raise ValueError("Input data must not be empty.")

        idx = obs.index

        prices = obs.to_numpy(dtype=float)
        n_steps, n_vars = prices.shape

        F = np.block(
            [
                [np.eye(n_vars), dt * np.eye(n_vars)],
                [np.zeros((n_vars, n_vars)), np.eye(n_vars)],
            ]
        )

        H = np.block(
            [
                np.eye(n_vars),
                np.zeros((n_vars, n_vars)),
            ]
        )

        Q = process_var * np.eye(2 * n_vars)
        R = measurement_var * np.eye(n_vars)

        x = np.hstack([prices[0], np.zeros(n_vars)])
        P = initial_cov * np.eye(2 * n_vars)

        estimates = np.full((n_steps, n_vars), np.nan, dtype=float)
        velocities = np.full((n_steps, n_vars), np.nan, dtype=float)

        for i in range(n_steps):
            z = prices[i]

            x_pred = F @ x
            P_pred = F @ P @ F.T + Q

            y = z - H @ x_pred
            S = H @ P_pred @ H.T + R
            K = P_pred @ H.T @ np.linalg.inv(S)

            x = x_pred + K @ y
            P = (np.eye(2 * n_vars) - K @ H) @ P_pred

            estimates[i, :] = x[:n_vars]
            velocities[i, :] = x[n_vars:]

        result = pd.DataFrame(estimates, index=idx, columns=columns)

        if return_velocity:
            vel_cols = [f"{col}_velocity" for col in columns]
            vel_df = pd.DataFrame(velocities, index=idx, columns=vel_cols)
            result = pd.concat([result, vel_df], axis=1)

        return result

    # *******************************************************************************************************************
    # Functionname:       KalmanFilters.filter_ohlc(data, open_col: str = "open", high_col: str = "high",
    #                              low_col: str = "low", close_col: str = "close",
    #                              process_var: float = 1e-5, measurement_var: float = 1e-1,
    #                              initial_cov: float = 1e-1, dt: float = 1.0,
    #                              return_velocity: bool = False)
    #
    # @brief              Filter OHLC columns while preserving canonical output order.
    # @pre                Data must contain selected OHLC columns
    # @post               Returns filtered OHLC DataFrame with optional velocity columns
    # @param[in]          data: Input tabular data
    #                     open_col: Open column name
    #                     high_col: High column name
    #                     low_col: Low column name
    #                     close_col: Close column name
    #                     process_var: Process noise variance
    #                     measurement_var: Measurement noise variance
    #                     initial_cov: Initial covariance scale
    #                     dt: Time step
    #                     return_velocity: Include velocity outputs when True
    # @param[out]         result: Filtered OHLC DataFrame
    #
    # @callsequence       @startuml
    #                     title KalmanFilters.filter_ohlc
    #                     start
    #                     :Map OHLC column names;
    #                     :Call filter_multivariate for selected columns;
    #                     if (return_velocity?) then (yes)
    #                       :Reorder OHLC + velocity columns;
    #                       :Return DataFrame;
    #                     else (no)
    #                       :Return OHLC columns only;
    #                     endif
    #                     end
    #                     @enduml
    # *******************************************************************************************************************
    @staticmethod
    def filter_ohlc(
        data,
        open_col: str = "open",
        high_col: str = "high",
        low_col: str = "low",
        close_col: str = "close",
        process_var: float = 1e-5,
        measurement_var: float = 1e-1,
        initial_cov: float = 1e-1,
        dt: float = 1.0,
        return_velocity: bool = False,
    ) -> pd.DataFrame:
        columns = [open_col, high_col, low_col, close_col]
        result = KalmanFilters.filter_multivariate(
            data=data,
            columns=columns,
            process_var=process_var,
            measurement_var=measurement_var,
            initial_cov=initial_cov,
            dt=dt,
            return_velocity=return_velocity,
        )

        if return_velocity:
            ordered_cols = columns + [f"{col}_velocity" for col in columns]
            return result[ordered_cols]

        return result[columns]


# ***********************************************************************************************************************
# Functionname:       kalman_filter_1d(series, process_var: float = 1e-5, measurement_var: float = 1e-1,
#                              initial_cov: float = 1e-1, dt: float = 1.0,
#                              return_velocity: bool = False)
#
# @brief              Backward-compatible wrapper for KalmanFilters.filter_1d.
# @pre                Same preconditions as KalmanFilters.filter_1d
# @post               Returns same output as KalmanFilters.filter_1d
# @param[in]          series: Input scalar sequence
#                     process_var: Process noise variance
#                     measurement_var: Measurement noise variance
#                     initial_cov: Initial covariance scale
#                     dt: Time step
#                     return_velocity: Include velocity output when True
# @param[out]         result: Filtered output as Series or DataFrame
# ***********************************************************************************************************************
def kalman_filter_1d(
    series,
    process_var: float = 1e-5,
    measurement_var: float = 1e-1,
    initial_cov: float = 1e-1,
    dt: float = 1.0,
    return_velocity: bool = False,
) -> pd.Series | pd.DataFrame:
    return KalmanFilters.filter_1d(
        series=series,
        process_var=process_var,
        measurement_var=measurement_var,
        initial_cov=initial_cov,
        dt=dt,
        return_velocity=return_velocity,
    )


# ***********************************************************************************************************************
# Functionname:       kalman_filter_multivariate(data: pd.DataFrame, columns: list[str] | None = None,
#                              process_var: float = 1e-5, measurement_var: float = 1e-1,
#                              initial_cov: float = 1e-1, dt: float = 1.0,
#                              return_velocity: bool = False)
#
# @brief              Backward-compatible wrapper for KalmanFilters.filter_multivariate.
# @pre                Same preconditions as KalmanFilters.filter_multivariate
# @post               Returns same output as KalmanFilters.filter_multivariate
# @param[in]          data: Input DataFrame
#                     columns: Columns to filter
#                     process_var: Process noise variance
#                     measurement_var: Measurement noise variance
#                     initial_cov: Initial covariance scale
#                     dt: Time step
#                     return_velocity: Include velocity outputs when True
# @param[out]         result: Filtered DataFrame
# ***********************************************************************************************************************
def kalman_filter_multivariate(
    data: pd.DataFrame,
    columns: list[str] | None = None,
    process_var: float = 1e-5,
    measurement_var: float = 1e-1,
    initial_cov: float = 1e-1,
    dt: float = 1.0,
    return_velocity: bool = False,
) -> pd.DataFrame:
    return KalmanFilters.filter_multivariate(
        data=data,
        columns=columns,
        process_var=process_var,
        measurement_var=measurement_var,
        initial_cov=initial_cov,
        dt=dt,
        return_velocity=return_velocity,
    )


# ***********************************************************************************************************************
# Functionname:       kalman_filter_ohlc(data, open_col: str = "open", high_col: str = "high",
#                              low_col: str = "low", close_col: str = "close",
#                              process_var: float = 1e-5, measurement_var: float = 1e-1,
#                              initial_cov: float = 1e-1, dt: float = 1.0,
#                              return_velocity: bool = False)
#
# @brief              Backward-compatible wrapper for KalmanFilters.filter_ohlc.
# @pre                Same preconditions as KalmanFilters.filter_ohlc
# @post               Returns same output as KalmanFilters.filter_ohlc
# @param[in]          data: Input tabular data
#                     open_col: Open column name
#                     high_col: High column name
#                     low_col: Low column name
#                     close_col: Close column name
#                     process_var: Process noise variance
#                     measurement_var: Measurement noise variance
#                     initial_cov: Initial covariance scale
#                     dt: Time step
#                     return_velocity: Include velocity outputs when True
# @param[out]         result: Filtered OHLC DataFrame
# ***********************************************************************************************************************
def kalman_filter_ohlc(
    data,
    open_col: str = "open",
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    process_var: float = 1e-5,
    measurement_var: float = 1e-1,
    initial_cov: float = 1e-1,
    dt: float = 1.0,
    return_velocity: bool = False,
) -> pd.DataFrame:
    return KalmanFilters.filter_ohlc(
        data=data,
        open_col=open_col,
        high_col=high_col,
        low_col=low_col,
        close_col=close_col,
        process_var=process_var,
        measurement_var=measurement_var,
        initial_cov=initial_cov,
        dt=dt,
        return_velocity=return_velocity,
    )


__all__ = [
    "KalmanFilters",
    "kalman_filter_1d",
    "kalman_filter_multivariate",
    "kalman_filter_ohlc",
]
