from locale import normalize
import numpy as np
import pickle
from collections import namedtuple
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import StandardScaler
from scipy.stats import gaussian_kde
from tqdm import tqdm
from ..nonconformist import (
    ClassifierAdapter,
    IcpClassifier,
    TcpClassifier,
    ClassifierNc,
    MarginErrFunc,
)


class PredictionSet(BaseEstimator, ClassifierMixin):
    """Class PredictionSet: Obtain prediction sets.

    Attributes:

        obj: an object;
            fitted object containing methods `fit` and `predict`

        method: a string;
            method for constructing the prediction sets.
            Currently "icp" (default, inductive conformal) and "tcp" (transductive conformal)

        level: a float;
            Confidence level for prediction sets. Default is None,
            95 is equivalent to a miscoverage error of 5 (%)

        seed: an integer;
            Reproducibility of fit (there's a random split between fitting and calibration data)
    """

    def __init__(
        self,
        obj,
        method="icp",
        level=None,
        seed=123,
    ):

        self.obj = obj
        self.method = method
        self.level = level
        self.seed = seed
        if self.level is not None:
            self.alpha_ = 1 - self.level / 100
        self.quantile_ = None
        self.icp_ = None
        self.tcp_ = None

        if self.method == "icp":
            self.icp_ = IcpClassifier(
                ClassifierNc(ClassifierAdapter(self.obj), MarginErrFunc()),
            )
        elif self.method == "tcp":
            self.tcp_ = TcpClassifier(
                ClassifierNc(ClassifierAdapter(self.obj), MarginErrFunc()),
            )
        else:
            raise ValueError("`self.method` must be in ('icp', 'tcp')")

    def fit(self, X, y, sample_weight=None, **kwargs):
        """Fit the `method` to training data (X, y).

        Args:

            X: array-like, shape = [n_samples, n_features];
                Training set vectors, where n_samples is the number
                of samples and n_features is the number of features.

            y: array-like, shape = [n_samples, ]; Target values.

            sample_weight: array-like, shape = [n_samples]
                Sample weights.

        """
        if self.method == "icp":

            X_train, X_calibration, y_train, y_calibration = train_test_split(
                X, y, test_size=0.5, random_state=self.seed
            )
            self.icp_.fit(X_train, y_train)
            self.icp_.calibrate(X_calibration, y_calibration)

        elif self.method == "tcp":

            self.tcp_.fit(X, y)

        return self

    def predict(self, X, **kwargs):
        """Obtain predictions and prediction sets

        Args:

            X: array-like, shape = [n_samples, n_features];
                Testing set vectors, where n_samples is the number
                of samples and n_features is the number of features.

        """

        if self.method == "icp":
            return self.icp_.predict(X, significance=self.alpha_, **kwargs)

        elif self.method == "tcp":
            return self.tcp_.predict(X, significance=self.alpha_, **kwargs)

        else:
            raise ValueError("`self.method` must be in ('icp', 'tcp')")

    def predict_proba(self, X):
        predictions = self.predict(X)
        return np.eye(len(np.unique(predictions)))[predictions]
