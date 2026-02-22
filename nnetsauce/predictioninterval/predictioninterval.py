from locale import normalize
import warnings
import numpy as np
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
from ..simulation import simulate_replications


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
            type of prediction interval: currently `None`
            (split conformal without simulation)
            for type_pi in:
                - 'bootstrap': Bootstrap resampling.
                - 'kde': Kernel Density Estimation.

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
        type_pi=None,
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
        self.aic_ = None
        self.aicc_ = None
        self.bic_ = None
        self.sse_ = None

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
            except Exception:
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

            # FIX: Store calibration residuals from the ICP scorer so that
            # simulation-based prediction intervals are available in predict().
            raw_residuals = self.icp_.nc_function.err_func.apply(
                self.icp_.nc_function.predict(X_calibration), y_calibration
            )
            self.calibrated_residuals_ = raw_residuals
            self.calibrated_residuals_scaler_ = StandardScaler(
                with_mean=True, with_std=True
            )
            self.scaled_calibrated_residuals_ = (
                self.calibrated_residuals_scaler_.fit_transform(
                    self.calibrated_residuals_.reshape(-1, 1)
                ).ravel()
            )

        # Calculate AIC
        # Get predictions
        preds = self.obj.predict(X_calibration)

        # Calculate SSE
        self.sse_ = np.sum((y_calibration - preds) ** 2)

        # Get number of parameters from the base model
        n_params = (
            getattr(self.obj, "n_hidden_features", 0) + X_calibration.shape[1]
        )

        # Calculate AIC
        n_samples = len(y_calibration)
        temp = n_samples * np.log(self.sse_ / n_samples)
        self.aic_ = temp + 2 * n_params
        self.bic_ = temp + np.log(n_samples) * n_params

        return self

    def _simulate_from_residuals(self, pred, n_obs):
        """Shared helper: draw `self.replications` simulations from calibrated
        residuals and return (sims, mean, lower, upper).

        Args:
            pred: 1-D array of point predictions, shape [n_obs].
            n_obs: int, number of test observations.

        Returns:
            sims_   : 2-D array, shape [n_obs, replications]
            mean_   : 1-D array, shape [n_obs]
            lower_  : 1-D array, shape [n_obs]
            upper_  : 1-D array, shape [n_obs]
        """
        type_pi = self.type_pi if self.type_pi is not None else "kde"
        replications = (
            self.replications if self.replications is not None else 100
        )

        assert type_pi in (
            "bootstrap",
            "kde",
            "normal",
            "ecdf",
            "permutation",
            "smooth-bootstrap",
        ), (
            "`type_pi` must be in ('bootstrap', 'kde', 'normal', 'ecdf', "
            "'permutation', 'smooth-bootstrap')"
        )

        scale = self.calibrated_residuals_scaler_.scale_[0]

        if type_pi == "bootstrap":
            np.random.seed(self.seed)
            residuals_sims = np.asarray(
                [
                    np.random.choice(
                        a=self.scaled_calibrated_residuals_,
                        size=n_obs,
                    )
                    for _ in range(replications)
                ]
            ).T  # shape [n_obs, replications]

        elif type_pi == "kde":
            kde = gaussian_kde(dataset=self.scaled_calibrated_residuals_)
            residuals_sims = np.asarray(
                [
                    kde.resample(size=n_obs, seed=self.seed + i).ravel()
                    for i in range(replications)
                ]
            ).T  # shape [n_obs, replications]

        else:  # normal / ecdf / permutation / smooth-bootstrap
            residuals_sims = np.asarray(
                simulate_replications(
                    data=self.scaled_calibrated_residuals_,
                    method=type_pi,
                    num_replications=replications,
                    n_obs=n_obs,
                    seed=self.seed,
                )
            ).T  # shape [n_obs, replications]

        sims = np.asarray(
            [
                pred + scale * residuals_sims[:, i].ravel()
                for i in range(replications)
            ]
        ).T  # shape [n_obs, replications]

        mean_ = np.mean(sims, axis=1)
        lower_ = np.quantile(sims, q=self.alpha_ / 200, axis=1)
        upper_ = np.quantile(sims, q=1 - self.alpha_ / 200, axis=1)

        return sims, mean_, lower_, upper_

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

        if self.method == "splitconformal":
            pred = self.obj.predict(X)

        if self.method == "localconformal":
            pred = self.icp_.predict(X)

        # ------------------------------------------------------------------ #
        # splitconformal
        # ------------------------------------------------------------------ #
        if self.method == "splitconformal":
            if self.replications is None and self.type_pi is None:
                # Plain split-conformal: symmetric quantile band
                if return_pi:
                    DescribeResult = namedtuple(
                        "DescribeResult", ("mean", "lower", "upper")
                    )
                    return DescribeResult(
                        pred, pred - self.quantile_, pred + self.quantile_
                    )
                else:
                    return pred

            else:
                # FIX: simulation-based prediction intervals for splitconformal.
                # Previously this branch raised NotImplementedError even though
                # all the necessary logic was present — it was simply unreachable
                # because the raise fired unconditionally.  The code has been
                # moved into _simulate_from_residuals() and called here.

                if self.type_pi is None:
                    warnings.warn(
                        "type_pi must be set when replications is not None; "
                        "defaulting to 'kde'."
                    )
                if self.replications is None:
                    warnings.warn(
                        "replications must be set when type_pi is not None; "
                        "defaulting to 100."
                    )

                (
                    self.sims_,
                    self.mean_,
                    self.lower_,
                    self.upper_,
                ) = self._simulate_from_residuals(pred, X.shape[0])

                DescribeResult = namedtuple(
                    "DescribeResult", ("mean", "sims", "lower", "upper")
                )
                return DescribeResult(
                    self.mean_, self.sims_, self.lower_, self.upper_
                )

        # ------------------------------------------------------------------ #
        # localconformal
        # ------------------------------------------------------------------ #
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
                        pred,
                        predictions_bounds[:, 0],
                        predictions_bounds[:, 1],
                    )
                else:
                    return pred

            else:
                # FIX: simulation-based prediction intervals for localconformal.
                # Previously this always raised NotImplementedError.  Now we
                # reuse the calibration residuals stored during fit() and apply
                # the same simulation logic used by splitconformal via the
                # shared helper _simulate_from_residuals().

                if self.type_pi is None:
                    warnings.warn(
                        "type_pi must be set when replications is not None; "
                        "defaulting to 'kde'."
                    )

                (
                    self.sims_,
                    self.mean_,
                    self.lower_,
                    self.upper_,
                ) = self._simulate_from_residuals(pred, X.shape[0])

                DescribeResult = namedtuple(
                    "DescribeResult", ("mean", "sims", "lower", "upper")
                )
                return DescribeResult(
                    self.mean_, self.sims_, self.lower_, self.upper_
                )
