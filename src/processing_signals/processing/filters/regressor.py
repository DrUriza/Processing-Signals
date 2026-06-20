
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures


# \file **********************************************************************
# COMPANY:            ELATIN
# PROJECT:            SIGNAL_ANALYSIS
# COMPONENT:          FILTERS
# MODULE NAME:        regressor.py
# DESCRIPTION:        @brief Polynomial regression utilities for close prices
# CREATION DATE:      26.04.2026
# VERSION:            $Revision: 0.2$
# CHANGES:            26.04.2026 - Refactored to class-based OOP with UML docs.
# *****************************************************************************

class RegressorFilters:
    # *******************************************************************************************************************
    # Functionname:       RegressorFilters.polynomial_close_fit(data: pd.DataFrame, max_degree: int = 15)
    #
    # @brief              Fit close prices with polynomial models and select the minimum MSE degree.
    # @pre                data is DataFrame with close column and max_degree > 0
    # @post               Returns best fitted signal and selected polynomial degree
    # @param[in]          data: Input DataFrame containing close prices
    #                     max_degree: Maximum polynomial degree to evaluate
    # @param[out]         result: Tuple(predicted_close, optimal_degree)
    #
    # @callsequence       @startuml
    #                     title RegressorFilters.polynomial_close_fit
    #                     start
    #                     :Validate input DataFrame and close column;
    #                     :Drop NaN close values;
    #                     :Build time axis [1..N];
    #                     :Initialize min_error = inf;
    #                     repeat
    #                       :Create polynomial features for degree d;
    #                       :Fit linear regression;
    #                       :Predict fitted series;
    #                       :Compute MSE;
    #                       if (MSE improves?) then (yes)
    #                         :Store optimal degree and prediction;
    #                       endif
    #                     repeat while (d <= max_degree)
    #                     :Return (optimal prediction, optimal degree);
    #                     end
    #                     @enduml
    # *******************************************************************************************************************
    @staticmethod
    def polynomial_close_fit(data: pd.DataFrame, max_degree: int = 15) -> tuple[np.ndarray, int]:
        if not isinstance(data, pd.DataFrame):
            raise ValueError("data must be a pandas DataFrame.")
        if "close" not in data.columns:
            raise ValueError("data must include a 'close' column.")
        if not isinstance(max_degree, int) or max_degree <= 0:
            raise ValueError("max_degree must be a positive integer.")

        close_values = data["close"].dropna(axis="rows").to_numpy(dtype=float)
        if close_values.size < 2:
            raise ValueError("close must contain at least 2 valid values.")

        time = np.arange(1, len(close_values) + 1, dtype=float).reshape(-1, 1)
        min_error = float("inf")
        optimal_degree = 1
        optimal_y_pred = close_values.copy()

        for degree in range(1, max_degree + 1):
            poly_reg = PolynomialFeatures(degree=degree)
            x_poly = poly_reg.fit_transform(time)
            model = LinearRegression().fit(x_poly, close_values)
            y_pred = model.predict(x_poly)
            error = mean_squared_error(close_values, y_pred)
            if error < min_error:
                min_error = error
                optimal_degree = degree
                optimal_y_pred = y_pred

        return optimal_y_pred, optimal_degree


# ***********************************************************************************************************************
# Functionname:       Regressor(data: pd.DataFrame)
#
# @brief              Backward-compatible wrapper for RegressorFilters.polynomial_close_fit.
# @pre                Same preconditions as RegressorFilters.polynomial_close_fit
# @post               Returns same output as RegressorFilters.polynomial_close_fit
# @param[in]          data: Input DataFrame containing close prices
# @param[out]         result: Tuple(predicted_close, optimal_degree)
# ***********************************************************************************************************************
def Regressor(data: pd.DataFrame) -> tuple[np.ndarray, int]:
    return RegressorFilters.polynomial_close_fit(data=data)


__all__ = ["RegressorFilters", "Regressor"]