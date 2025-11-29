# Author: @thierrymoudiki
# Date: 2025-01-03

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy

from collections import namedtuple
from sklearn.base import BaseEstimator, RegressorMixin
from scipy.optimize import differential_evolution
from dataclasses import dataclass
from typing import Dict, Any, List
from sklearn.base import clone
from sklearn.base import BaseEstimator, RegressorMixin
from scipy.optimize import minimize, differential_evolution
from sklearn.ensemble import RandomForestRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import ElasticNet
from sklearn.datasets import load_diabetes, fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from warnings import filterwarnings

filterwarnings("ignore")


@dataclass
class QuantileResult:
    params: Dict[str, Any]
    predictions: np.ndarray
    loss: float
    iterations: int


class QuantileRegressor(BaseEstimator, RegressorMixin):
    """
    Quantile Regressor.

    Parameters:

        obj: base model (regression model)
            The base regressor from which to build a
            quantile regressor.

        level: int, default=95
            The level of the quantiles to compute.

        scoring: str, default="predictions"
            The scoring to use for the optimization and constructing
            prediction intervals (predictions, residuals, conformal,
              studentized, conformal-studentized).

    Attributes:

        obj_ : base model (regression model)
            The base regressor from which to build a
            quantile regressor.

        offset_multipliers_ : list
            The multipliers for the offset.

        scoring_residuals_ : list
            The residuals for the scoring.

        student_multiplier_ : float
            The multiplier for the student.

    """

    def __init__(self, obj, level=95, scoring="predictions"):
        assert scoring in (
            "predictions",
            "residuals",
            "conformal",
            "studentized",
            "conformal-studentized",
        ), "scoring must be 'predictions' or 'residuals'"
        self.obj = obj
        low_risk_level = (1 - level / 100) / 2
        self.quantiles = [low_risk_level, 0.5, 1 - low_risk_level]
        self.scoring = scoring
        self.offset_multipliers_ = None
        self.obj_ = None
        self.scoring_residuals_ = None
        self.student_multiplier_ = None

    def _compute_quantile_loss(self, residuals, quantile):
        """
        Compute the quantile loss for a given set of residuals and quantile.
        """
        return np.mean(
            residuals
            * (quantile * (residuals >= 0) + (quantile - 1) * (residuals < 0))
        )

    def _optimize_multiplier(
        self,
        y,
        base_predictions,
        prev_predictions,
        scoring_residuals=None,
        quantile=0.5,
    ):
        """
        Optimize the multiplier for a given quantile.
        """
        if not 0 < quantile < 1:
            raise ValueError("Quantile should be between 0 and 1.")

        n = len(y)

        def objective(log_multiplier):
            """
            Objective function for optimization.
            """
            # Convert to positive multiplier using exp
            multiplier = np.exp(log_multiplier[0])
            if self.scoring == "predictions":
                assert (
                    base_predictions is not None
                ), "base_predictions must be not None"
                # Calculate predictions
                if prev_predictions is None:
                    # For first quantile, subtract from conditional expectation
                    predictions = base_predictions - multiplier * np.abs(
                        base_predictions
                    )
                else:
                    # For other quantiles, add to previous quantile
                    offset = multiplier * np.abs(base_predictions)
                    predictions = prev_predictions + offset
            elif self.scoring in ("residuals", "conformal"):
                assert (
                    scoring_residuals is not None
                ), "scoring_residuals must be not None"
                # print("scoring_residuals", scoring_residuals)
                # Calculate predictions
                if prev_predictions is None:
                    # For first quantile, subtract from conditional expectation
                    predictions = base_predictions - multiplier * np.std(
                        scoring_residuals
                    ) / np.sqrt(len(scoring_residuals))
                    # print("predictions", predictions)
                else:
                    # For other quantiles, add to previous quantile
                    offset = (
                        multiplier
                        * np.std(scoring_residuals)
                        / np.sqrt(len(scoring_residuals))
                    )
                    predictions = prev_predictions + offset
            elif self.scoring in ("studentized", "conformal-studentized"):
                assert (
                    scoring_residuals is not None
                ), "scoring_residuals must be not None"
                # Calculate predictions
                if prev_predictions is None:
                    # For first quantile, subtract from conditional expectation
                    predictions = (
                        base_predictions - multiplier * self.student_multiplier_
                    )
                    # print("predictions", predictions)
                else:
                    # For other quantiles, add to previous quantile
                    offset = multiplier * self.student_multiplier_
                    predictions = prev_predictions + offset
            else:
                raise ValueError("Invalid argument 'scoring'")

            return self._compute_quantile_loss(y - predictions, quantile)

        # Optimize in log space for numerical stability
        # bounds = [(-10, 10)]  # log space bounds
        bounds = [(-100, 100)]  # log space bounds
        result = differential_evolution(
            objective,
            bounds,
            # popsize=15,
            # maxiter=100,
            # tol=1e-4,
            popsize=25,
            maxiter=200,
            tol=1e-6,
            disp=False,
        )

        return np.exp(result.x[0])

    def fit(self, X, y):
        """Fit the model to the data.

        Parameters:

            X: {array-like}, shape = [n_samples, n_features]
                Training vectors, where n_samples is the number of samples and
                n_features is the number of features.
            y: array-like, shape = [n_samples]
                Target values.
        """
        self.obj_ = clone(self.obj)

        if self.scoring in ("predictions", "residuals"):
            self.obj_.fit(X, y)
            base_predictions = self.obj_.predict(X)
            scoring_residuals = y - base_predictions
            self.scoring_residuals_ = scoring_residuals

        elif self.scoring == "conformal":
            X_train, X_calib, y_train, y_calib = train_test_split(
                X, y, test_size=0.5, random_state=42
            )
            self.obj_.fit(X_train, y_train)
            scoring_residuals = y_calib - self.obj_.predict(
                X_calib
            )  # These are calibration predictions
            self.scoring_residuals_ = scoring_residuals
            # Update base_predictions to use training predictions for optimization
            self.obj_.fit(X_calib, y_calib)
            base_predictions = self.obj_.predict(X_calib)

        elif self.scoring in ("studentized", "conformal-studentized"):
            # Calculate student multiplier
            if self.scoring == "conformal-studentized":
                X_train, X_calib, y_train, y_calib = train_test_split(
                    X, y, test_size=0.5, random_state=42
                )
                self.obj_.fit(X_train, y_train)
                scoring_residuals = y_calib - self.obj_.predict(X_calib)
                # Calculate studentized multiplier using calibration data
                self.student_multiplier_ = np.std(y_calib, ddof=1) / np.sqrt(
                    len(y_calib) - 1
                )
                self.obj_.fit(X_calib, y_calib)
                base_predictions = self.obj_.predict(X_calib)
            else:  # regular studentized
                self.obj_.fit(X, y)
                base_predictions = self.obj_.predict(X)
                scoring_residuals = y - base_predictions
                self.student_multiplier_ = np.std(y, ddof=1) / np.sqrt(
                    len(y) - 1
                )

        # Initialize storage for multipliers
        self.offset_multipliers_ = []
        # Keep track of current predictions for each quantile
        current_predictions = None

        # Fit each quantile sequentially
        for i, quantile in enumerate(self.quantiles):
            if self.scoring == "predictions":
                multiplier = self._optimize_multiplier(
                    y=y,
                    base_predictions=base_predictions,
                    prev_predictions=current_predictions,
                    quantile=quantile,
                )

                self.offset_multipliers_.append(multiplier)

                # Update current predictions
                if current_predictions is None:
                    # First quantile (lowest)
                    current_predictions = (
                        base_predictions - multiplier * np.abs(base_predictions)
                    )
                else:
                    # Subsequent quantiles
                    offset = multiplier * np.abs(base_predictions)
                    current_predictions = current_predictions + offset

            elif self.scoring == "residuals":
                multiplier = self._optimize_multiplier(
                    y=y,
                    base_predictions=base_predictions,
                    scoring_residuals=scoring_residuals,
                    prev_predictions=current_predictions,
                    quantile=quantile,
                )

                self.offset_multipliers_.append(multiplier)

                # Update current predictions
                if current_predictions is None:
                    # First quantile (lowest)
                    current_predictions = (
                        base_predictions
                        - multiplier
                        * np.std(scoring_residuals)
                        / np.sqrt(len(scoring_residuals))
                    )
                else:
                    # Subsequent quantiles
                    offset = (
                        multiplier
                        * np.std(scoring_residuals)
                        / np.sqrt(len(scoring_residuals))
                    )
                    current_predictions = current_predictions + offset

            elif self.scoring == "conformal":
                multiplier = self._optimize_multiplier(
                    y=y_calib,
                    base_predictions=base_predictions,
                    scoring_residuals=scoring_residuals,
                    prev_predictions=current_predictions,
                    quantile=quantile,
                )

                self.offset_multipliers_.append(multiplier)

                # Update current predictions
                if current_predictions is None:
                    # First quantile (lowest)
                    current_predictions = (
                        base_predictions
                        - multiplier
                        * np.std(scoring_residuals)
                        / np.sqrt(len(scoring_residuals))
                    )
                else:
                    # Subsequent quantiles
                    offset = (
                        multiplier
                        * np.std(scoring_residuals)
                        / np.sqrt(len(scoring_residuals))
                    )
                    current_predictions = current_predictions + offset

            elif self.scoring in ("studentized", "conformal-studentized"):
                multiplier = self._optimize_multiplier(
                    y=y_calib if self.scoring == "conformal-studentized" else y,
                    base_predictions=base_predictions,
                    scoring_residuals=scoring_residuals,
                    prev_predictions=current_predictions,
                    quantile=quantile,
                )

                self.offset_multipliers_.append(multiplier)

                # Update current predictions
                if current_predictions is None:
                    current_predictions = (
                        base_predictions - multiplier * self.student_multiplier_
                    )
                else:
                    offset = multiplier * self.student_multiplier_
                    current_predictions = current_predictions + offset

        return self

    def predict(self, X, return_pi=False):
        """Predict the target variable.

        Parameters:

            X: {array-like}, shape = [n_samples, n_features]
                Training vectors, where n_samples is the number of samples and
                n_features is the number of features.

            return_pi: bool, default=True
                Whether to return the prediction intervals.
        """
        if self.obj_ is None or self.offset_multipliers_ is None:
            raise ValueError("Model not fitted yet.")

        base_predictions = self.obj_.predict(X)
        all_predictions = []

        if self.scoring == "predictions":
            # Generate first quantile
            current_predictions = base_predictions - self.offset_multipliers_[
                0
            ] * np.abs(base_predictions)
            all_predictions.append(current_predictions)

            # Generate remaining quantiles
            for multiplier in self.offset_multipliers_[1:]:
                offset = multiplier * np.abs(base_predictions)
                current_predictions = current_predictions + offset
                all_predictions.append(current_predictions)

        elif self.scoring in ("residuals", "conformal"):
            # Generate first quantile
            current_predictions = base_predictions - self.offset_multipliers_[
                0
            ] * np.std(self.scoring_residuals_) / np.sqrt(
                len(self.scoring_residuals_)
            )
            all_predictions.append(current_predictions)

            # Generate remaining quantiles
            for multiplier in self.offset_multipliers_[1:]:
                offset = (
                    multiplier
                    * np.std(self.scoring_residuals_)
                    / np.sqrt(len(self.scoring_residuals_))
                )
                current_predictions = current_predictions + offset
                all_predictions.append(current_predictions)

        elif self.scoring in ("studentized", "conformal-studentized"):
            # Generate first quantile
            current_predictions = (
                base_predictions
                - self.offset_multipliers_[0] * self.student_multiplier_
            )
            all_predictions.append(current_predictions)

            # Generate remaining quantiles
            for multiplier in self.offset_multipliers_[1:]:
                offset = multiplier * self.student_multiplier_
                current_predictions = current_predictions + offset
                all_predictions.append(current_predictions)

        if return_pi == False:
            return np.asarray(all_predictions[1])

        DescribeResult = namedtuple(
            "DecribeResult", ["mean", "lower", "upper", "median"]
        )
        DescribeResult.mean = base_predictions
        DescribeResult.lower = np.asarray(all_predictions[0])
        DescribeResult.median = np.asarray(all_predictions[1])
        DescribeResult.upper = np.asarray(all_predictions[2])

        return DescribeResult
