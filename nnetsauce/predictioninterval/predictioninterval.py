from locale import normalize
import numpy as np
import pickle
from collections import namedtuple
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
from scipy.stats import gaussian_kde
from tqdm import tqdm
from ..nonconformist import IcpRegressor
from ..nonconformist import RegressorNc
from ..nonconformist import RegressorNormalizer, AbsErrorErrFunc
from ..utils import Progbar


class PredictionInterval(BaseEstimator, RegressorMixin):
    """Class PredictionInterval: Obtain prediction intervals.

    Attributes:

        obj: an object;
            fitted object containing methods `fit` and `predict`

        method: a string;
            method for constructing the prediction intervals.
            Currently "splitconformal" (default) and "localconformal"

        level: a float;
            Confidence level for prediction intervals. Default is 95,
            equivalent to a miscoverage error of 5 (%)

        replications: an integer;
            Number of replications for simulated conformal (default is `None`)

        type_pi: a string;
            type of prediction interval: currently "kde" (default) or "bootstrap"

        type_split: a string;
            "random" (random split of data) or "sequential" (sequential split of data)

        seed: an integer;
            Reproducibility of fit (there's a random split between fitting and calibration data)
    """

    def __init__(
        self,
        obj,
        method="splitconformal",
        level=95,
        type_pi="bootstrap",
        type_split="random",
        replications=None,
        kernel=None,
        agg="mean",
        seed=123,
    ):

        self.obj = obj
        self.method = method
        self.level = level
        self.type_pi = type_pi
        self.type_split = type_split
        self.replications = replications
        self.kernel = kernel
        self.agg = agg
        self.seed = seed
        self.alpha_ = 1 - self.level / 100
        self.quantile_ = None
        self.icp_ = None
        self.calibrated_residuals_ = None
        self.scaled_calibrated_residuals_ = None
        self.calibrated_residuals_scaler_ = None
        self.kde_ = None

    def fit(self, X, y):
        """Fit the `method` to training data (X, y).

        Args:

            X: array-like, shape = [n_samples, n_features];
                Training set vectors, where n_samples is the number
                of samples and n_features is the number of features.

            y: array-like, shape = [n_samples, ]; Target values.

        """

        if self.type_split == "random":
            X_train, X_calibration, y_train, y_calibration = train_test_split(
                X, y, test_size=0.5, random_state=self.seed
            )
        elif self.type_split == "sequential":
            n_x = X.shape[0]
            n_x_half = n_x // 2
            first_half_idx = range(0, n_x_half)
            second_half_idx = range(n_x_half, n_x)
            X_train = X[first_half_idx, :]
            X_calibration = X[second_half_idx, :]
            y_train = y[first_half_idx]
            y_calibration = y[second_half_idx]

        if self.method == "splitconformal":

            n_samples_calibration = X_calibration.shape[0]
            self.obj.fit(X_train, y_train)
            preds_calibration = self.obj.predict(X_calibration)
            self.calibrated_residuals_ = y_calibration - preds_calibration
            absolute_residuals = np.abs(self.calibrated_residuals_)
            self.calibrated_residuals_scaler_ = StandardScaler(
                with_mean=True, with_std=True
            )
            self.scaled_calibrated_residuals_ = (
                self.calibrated_residuals_scaler_.fit_transform(
                    self.calibrated_residuals_.reshape(-1, 1)
                ).ravel()
            )
            try:
                # numpy version >= 1.22
                self.quantile_ = np.quantile(
                    a=absolute_residuals, q=self.level / 100, method="higher"
                )
            except:
                # numpy version < 1.22
                self.quantile_ = np.quantile(
                    a=absolute_residuals,
                    q=self.level / 100,
                    interpolation="higher",
                )

        if self.method == "localconformal":

            mad_estimator = ExtraTreesRegressor()
            normalizer = RegressorNormalizer(
                self.obj, mad_estimator, AbsErrorErrFunc()
            )
            nc = RegressorNc(self.obj, AbsErrorErrFunc(), normalizer)
            self.icp_ = IcpRegressor(nc)
            self.icp_.fit(X_train, y_train)
            self.icp_.calibrate(X_calibration, y_calibration)
            preds_calibration = self.icp_.predict(X_calibration)
            self.calibrated_residuals_ = y_calibration - preds_calibration
            absolute_residuals = np.abs(self.calibrated_residuals_)
            self.calibrated_residuals_scaler_ = StandardScaler(
                with_mean=True, with_std=True
            )
            self.scaled_calibrated_residuals_ = (
                self.calibrated_residuals_scaler_.fit_transform(
                    self.calibrated_residuals_.reshape(-1, 1)
                ).ravel()
            )

        return self

    def predict(self, X, return_pi=False):
        """Obtain predictions and prediction intervals

        Args:

            X: array-like, shape = [n_samples, n_features];
                Testing set vectors, where n_samples is the number
                of samples and n_features is the number of features.

            return_pi: boolean
                Whether the prediction interval is returned or not.
                Default is False, for compatibility with other _estimators_.
                If True, a tuple containing the predictions + lower and upper
                bounds is returned.

        """

        pred = self.obj.predict(X)

        if self.method == "splitconformal":

            if self.replications is None:

                if return_pi:

                    DescribeResult = namedtuple(
                        "DescribeResult", ("mean", "lower", "upper")
                    )

                    return DescribeResult(
                        pred, pred - self.quantile_, pred + self.quantile_
                    )

                else:

                    return pred

            else:  # if self.replications is not None

                assert self.type_pi in (
                    "bootstrap",
                    "kde",
                ), "`self.type_pi` must be in ('bootstrap', 'kde')"

                if self.type_pi == "bootstrap":
                    np.random.seed(self.seed)
                    self.residuals_sims_ = np.asarray(
                        [
                            np.random.choice(
                                a=self.scaled_calibrated_residuals_,
                                size=X.shape[0],
                            )
                            for _ in range(self.replications)
                        ]
                    ).T
                    self.sims_ = np.asarray(
                        [
                            pred
                            + self.calibrated_residuals_scaler_.scale_[0]
                            * self.residuals_sims_[:, i].ravel()
                            for i in range(self.replications)
                        ]
                    ).T
                elif self.type_pi == "kde":
                    self.kde_ = gaussian_kde(
                        dataset=self.scaled_calibrated_residuals_
                    )
                    self.sims_ = np.asarray(
                        [
                            pred
                            + self.calibrated_residuals_scaler_.scale_[0]
                            * self.kde_.resample(
                                size=X.shape[0], seed=self.seed + i
                            ).ravel()
                            for i in range(self.replications)
                        ]
                    ).T

                self.mean_ = np.mean(self.sims_, axis=1)
                self.lower_ = np.quantile(
                    self.sims_, q=self.alpha_ / 200, axis=1
                )
                self.upper_ = np.quantile(
                    self.sims_, q=1 - self.alpha_ / 200, axis=1
                )

                DescribeResult = namedtuple(
                    "DescribeResult", ("mean", "sims", "lower", "upper")
                )

                return DescribeResult(
                    self.mean_, self.sims_, self.lower_, self.upper_
                )

        if self.method == "localconformal":

            if self.replications is None:

                if return_pi:

                    predictions_bounds = self.icp_.predict(
                        X, significance=1 - self.level
                    )
                    DescribeResult = namedtuple(
                        "DescribeResult", ("mean", "lower", "upper")
                    )
                    return DescribeResult(
                        pred, predictions_bounds[:, 0], predictions_bounds[:, 1]
                    )

                else:

                    return pred

            else:  # if self.replications is not None

                assert self.type_pi in (
                    "bootstrap",
                    "kde",
                ), "`self.type_pi` must be in ('bootstrap', 'kde')"

                if self.type_pi == "bootstrap":
                    np.random.seed(self.seed)
                    self.residuals_sims_ = np.asarray(
                        [
                            np.random.choice(
                                a=self.scaled_calibrated_residuals_,
                                size=X.shape[0],
                            )
                            for _ in range(self.replications)
                        ]
                    ).T
                    self.sims_ = np.asarray(
                        [
                            pred
                            + self.calibrated_residuals_scaler_.scale_[0]
                            * self.residuals_sims_[:, i].ravel()
                            for i in tqdm(range(self.replications))
                        ]
                    ).T
                elif self.type_pi == "kde":
                    self.kde_ = gaussian_kde(
                        dataset=self.scaled_calibrated_residuals_
                    )
                    self.sims_ = np.asarray(
                        [
                            pred
                            + self.calibrated_residuals_scaler_.scale_[0]
                            * self.kde_.resample(
                                size=X.shape[0], seed=self.seed + i
                            ).ravel()
                            for i in tqdm(range(self.replications))
                        ]
                    ).T

                self.mean_ = np.mean(self.sims_, axis=1)
                self.lower_ = np.quantile(
                    self.sims_, q=self.alpha_ / 200, axis=1
                )
                self.upper_ = np.quantile(
                    self.sims_, q=1 - self.alpha_ / 200, axis=1
                )

                DescribeResult = namedtuple(
                    "DescribeResult", ("mean", "sims", "lower", "upper")
                )

                return DescribeResult(
                    self.mean_, self.sims_, self.lower_, self.upper_
                )
