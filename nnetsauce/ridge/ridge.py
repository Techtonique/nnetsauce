# Authors: Thierry Moudiki
#
# License: BSD 3 Clear

import numpy as np
import platform
import warnings
from sklearn.base import BaseEstimator
from sklearn.base import RegressorMixin
from numpy.linalg import inv
from ..utils import matrixops as mo

from .utils import get_beta

try:
    import jax.numpy as jnp
    from jax import device_put
    from jax.numpy.linalg import inv as jinv
except ImportError:
    pass


class RidgeRegressor(BaseEstimator, RegressorMixin):
    """Ridge.

    Attributes:

        reg_lambda: float
            regularization parameter.

        backend: str
            type of backend; must be in ('cpu', 'gpu', 'tpu')

    """

    def __init__(self, reg_lambda=0.1, backend="cpu"):
        assert backend in (
            "cpu",
            "gpu",
            "tpu",
        ), "`backend` must be in ('cpu', 'gpu', 'tpu')"

        sys_platform = platform.system()

        if (sys_platform == "Windows") and (backend in ("gpu", "tpu")):
            warnings.warn(
                "No GPU/TPU computing on Windows yet, backend set to 'cpu'"
            )
            backend = "cpu"

        self.reg_lambda = reg_lambda
        self.backend = backend
        self.coef_ = None

    def fit(self, X, y, **kwargs):
        """Fit matrixops (classifier) to training data (X, y)

        Args:

            X: {array-like}, shape = [n_samples, n_features]
                Training vectors, where n_samples is the number
                of samples and n_features is the number of features.

            y: array-like, shape = [n_samples]
                Target values.

            **kwargs: additional parameters to be passed to self.cook_training_set.

        Returns:

            self: object.

        """
        self.ym, centered_y = mo.center_response(y)
        self.xm = X.mean(axis=0)
        self.xsd = X.std(axis=0)
        self.xsd[self.xsd == 0] = 1  # avoid division by zero
        X_ = (X - self.xm[None, :]) / self.xsd[None, :]

        if self.backend == "cpu":
            if len(centered_y.shape) <= 1:
                eye_term = np.sqrt(self.reg_lambda) * np.eye(X.shape[1])
                X_ = np.row_stack((X_, eye_term))
                y_ = np.concatenate((centered_y, np.zeros(X.shape[1])))
                beta_info = get_beta(X_, y_)
                self.coef_ = beta_info[0]
            else:
                try:
                    eye_term = np.sqrt(self.reg_lambda) * np.eye(X.shape[1])
                    X_ = np.row_stack((X_, eye_term))
                    y_ = np.row_stack(
                        (
                            centered_y,
                            np.zeros((eye_term.shape[0], centered_y.shape[1])),
                        )
                    )
                    beta_info = get_beta(X_, y_)
                    self.coef_ = beta_info[0]
                except Exception:
                    x = inv(
                        mo.crossprod(X_) + self.reg_lambda * np.eye(X_.shape[1])
                    )
                    hat_matrix = mo.tcrossprod(x, X_)
                    self.coef_ = mo.safe_sparse_dot(hat_matrix, centered_y)
            return self

        x = jinv(
            mo.crossprod(X_, backend=self.backend)
            + self.reg_lambda * jnp.eye(X_.shape[1])
        )

        hat_matrix = mo.tcrossprod(x, X_, backend=self.backend)
        self.coef_ = mo.safe_sparse_dot(
            hat_matrix, centered_y, backend=self.backend
        )
        return self

    def predict(self, X, **kwargs):
        """Predict test data X.

        Args:

            X: {array-like}, shape = [n_samples, n_features]
                Training vectors, where n_samples is the number
                of samples and n_features is the number of features.

            **kwargs: additional parameters to be passed to `predict_proba`

        Returns:

            model predictions: {array-like}

        """
        X_ = (X - self.xm[None, :]) / self.xsd[None, :]

        if self.backend == "cpu":
            if isinstance(self.ym, float):
                return self.ym + mo.safe_sparse_dot(X_, self.coef_)
            return self.ym[None, :] + mo.safe_sparse_dot(X_, self.coef_)

        # if self.backend in ("gpu", "tpu"):
        if isinstance(self.ym, float):
            return self.ym + mo.safe_sparse_dot(
                X_, self.coef_, backend=self.backend
            )
        return self.ym[None, :] + mo.safe_sparse_dot(
            X_, self.coef_, backend=self.backend
        )
