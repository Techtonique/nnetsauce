# Author: @thierrymoudiki
# Date: 2025-01-09

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy

from collections import namedtuple
from sklearn.base import BaseEstimator, ClassifierMixin
from scipy.optimize import differential_evolution
from dataclasses import dataclass
from functools import partial
from typing import Dict, Any, List
from sklearn.base import clone
from sklearn.base import BaseEstimator, ClassifierMixin
from scipy.optimize import minimize, differential_evolution
from sklearn.ensemble import RandomForestRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import ElasticNet
from sklearn.datasets import load_diabetes, fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from warnings import filterwarnings

from .quantileregression import QuantileRegressor
from nnetsauce.multitask.simplemultitaskClassifier import (
    SimpleMultitaskClassifier,
)

filterwarnings("ignore")


@dataclass
class QuantileResult:
    params: Dict[str, Any]
    predictions: np.ndarray
    loss: float
    iterations: int


class QuantileClassifier(BaseEstimator, ClassifierMixin):
    """
    Quantile Classifier.

    Parameters:

        obj: base model (classification model)
            The base classifier from which to build a
            quantile classifier.

        level: int, default=95
            The level of the quantiles to compute.

        scoring: str, default="predictions"
            The scoring to use for the optimization and constructing
            prediction intervals (predictions, residuals, conformal,
              studentized, conformal-studentized).

    Attributes:

        obj_ : base model (classification model)
            The base classifier from which to build a
            quantile classifier.

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
        quantileregressor = QuantileRegressor(self.obj)
        quantileregressor.predict = partial(
            quantileregressor.predict, return_pi=False
        )
        self.obj_ = SimpleMultitaskClassifier(quantileregressor)

    def fit(self, X, y, **kwargs):
        self.obj_.fit(X, y, **kwargs)

    def predict(self, X, **kwargs):
        return self.obj_.predict(X, **kwargs)

    def predict_proba(self, X, **kwargs):
        return self.obj_.predict_proba(X, **kwargs)
