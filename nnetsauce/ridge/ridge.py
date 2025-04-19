# Authors: Thierry Moudiki
#
# License: BSD 3 Clear

import numpy as np
import warnings
from sklearn.base import RegressorMixin
from ..base import Base

try:
    import jax.numpy as jnp
    import jax.nn as jnn
except ImportError:
    pass

from ..utils import matrixops as mo


class RidgeRegressor(Base, RegressorMixin):
    """Basic Ridge Regression model.

    Parameters:
        lambda_: float or array-like
            Ridge regularization parameter(s). Default is 0.
    """

    def __init__(
        self,
        lambda_=0.0,
        n_hidden_features=0,
        activation_name="relu",
        a=0.01,
        nodes_sim="sobol",
        bias=True,
        dropout=0,
        direct_link=True,
        n_clusters=0,
        cluster_encode=True,
        type_clust="kmeans",
        type_scaling=("std", "std", "std"),
        col_sample=1,
        row_sample=1,
        seed=123,
        backend="cpu",
    ):
        super().__init__(
            n_hidden_features=n_hidden_features,
            activation_name=activation_name,
            a=a,
            nodes_sim=nodes_sim,
            bias=bias,
            dropout=dropout,
            direct_link=direct_link,
            n_clusters=n_clusters,
            cluster_encode=cluster_encode,
            type_clust=type_clust,
            type_scaling=type_scaling,
            col_sample=col_sample,
            row_sample=row_sample,
            seed=seed,
            backend=backend,
        )
        self.lambda_ = lambda_

    def _center_scale_xy(self, X, y):
        """Center X and y, scale X."""
        n = X.shape[0]

        # Center X and y
        X_mean = np.mean(X, axis=0)
        y_mean = np.mean(y)
        X_centered = X - X_mean
        y_centered = y - y_mean

        # Scale X
        X_scale = np.sqrt(np.sum(X_centered**2, axis=0) / n)
        # Avoid division by zero
        X_scale = np.where(X_scale == 0, 1.0, X_scale)
        X_scaled = X_centered / X_scale

        return X_scaled, y_centered, X_mean, y_mean, X_scale

    def fit(self, X, y):
        """Fit Ridge regression model.

        Parameters:
            X : array-like of shape (n_samples, n_features)
                Training data
            y : array-like of shape (n_samples,)
                Target values

        Returns:
            self : returns an instance of self.
        """
        # Ensure numpy arrays
        X = np.asarray(X)
        y = np.asarray(y)
        print(f"\nInput shapes - X: {X.shape}, y: {y.shape}")
        print(f"First few X values: {X[:2]}")
        print(f"First few y values: {y[:2]}")

        if y.ndim == 2:
            y = y.ravel()

        # Center and scale
        X_scaled, y_centered, self.X_mean_, self.y_mean_, self.X_scale_ = (
            self._center_scale_xy(X, y)
        )

        # SVD decomposition
        U, d, Vt = np.linalg.svd(X_scaled, full_matrices=False)

        # Compute coefficients
        rhs = np.dot(U.T, y_centered)
        d2 = d**2

        print(f"d2 shape: {d2.shape}")
        print(f"rhs shape: {rhs.shape}")
        print(f"Vt shape: {Vt.shape}")

        if np.isscalar(self.lambda_):
            div = d2 + self.lambda_
            a = (d * rhs) / div
            print(f"\nSingle lambda case:")
            print(f"lambda: {self.lambda_}")
            print(f"div shape: {div.shape}")
            print(f"a shape: {a.shape}")
            self.coef_ = np.dot(Vt.T, a) / self.X_scale_
            print(f"coef shape: {self.coef_.shape}")
        else:
            coefs = []
            print(f"\nMultiple lambda case:")
            for lambda_ in self.lambda_:
                print(f"lambda: {lambda_}")
                div = d2 + lambda_
                print(f"div shape: {div.shape}")
                a = (d * rhs) / div
                print(f"a shape: {a.shape}")
                coef = np.dot(Vt.T, a) / self.X_scale_
                print(f"coef shape: {coef.shape}")
                coefs.append(coef)
            self.coef_ = np.array(coefs).T
            print(f"final coefs shape: {self.coef_.shape}")

        # Compute GCV, HKB and LW criteria
        y_pred = self.predict(X)
        try:
            resid = y - y_pred
        except Exception as e:
            resid = y[:, np.newaxis] - y_pred
        n, p = X.shape
        if resid.ndim == 1:
            s2 = np.sum(resid**2) / (n - p)
        else:
            s2 = np.sum(resid**2, axis=0) / (n - p)

        self.HKB_ = (p - 2) * s2 / np.sum(self.coef_**2)
        self.LW_ = (p - 2) * s2 * n / np.sum(y_pred**2)

        if np.isscalar(self.lambda_):
            div = d2 + self.lambda_
            self.GCV_ = np.sum((y - y_pred) ** 2) / (n - np.sum(d2 / div)) ** 2
        else:
            self.GCV_ = []
            for lambda_ in self.lambda_:
                div = d2 + lambda_
                try:
                    gcv = np.sum((y - y_pred) ** 2) / (n - np.sum(d2 / div)) ** 2
                except Exception as e:
                    gcv = (
                        np.sum((y[:, np.newaxis] - y_pred) ** 2)
                        / (n - np.sum(d2 / div)) ** 2
                    )
                self.GCV_.append(gcv)
            self.GCV_ = np.array(self.GCV_)

        return self

    def predict(self, X):
        """Predict using the Ridge regression model.

        Parameters:
            X : array-like of shape (n_samples, n_features)
                Samples to predict for

        Returns:
            y_pred : array-like of shape (n_samples,)
                Returns predicted values.
        """
        X = self.cook_test_set(X)

        if self.backend == "cpu":
            if np.isscalar(self.lambda_):
                return (
                    mo.safe_sparse_dot(X, self.coef_, backend=self.backend)
                    + self.y_mean_
                )
            else:
                return jnp.array(
                    [
                        mo.safe_sparse_dot(X, coef, backend=self.backend) + self.y_mean_
                        for coef in self.coef_.T
                    ]
                ).T
        else:
            if np.isscalar(self.lambda_):
                return (
                    mo.safe_sparse_dot(X, self.coef_, backend=self.backend)
                    + self.y_mean_
                )
            else:
                return jnp.array(
                    [
                        mo.safe_sparse_dot(X, coef, backend=self.backend) + self.y_mean_
                        for coef in self.coef_.T
                    ]
                ).T

    def decision_function(self, X):
        """Compute the decision function of X.

        Parameters:
            X : array-like of shape (n_samples, n_features)
                Samples

        Returns:
            decision : array-like of shape (n_samples,) or (n_samples, n_lambdas)
                Decision function of the input samples. The order of outputs is the same
                as that of the provided lambda_ values. For a single lambda, returns
                array of shape (n_samples,). For multiple lambdas, returns array of shape
                (n_samples, n_lambdas).
        """
        X = self.cook_test_set(X)

        if self.backend == "cpu":
            if np.isscalar(self.lambda_):
                return mo.safe_sparse_dot(X, self.coef_, backend=self.backend)
            else:
                return np.array(
                    [
                        mo.safe_sparse_dot(X, coef, backend=self.backend)
                        for coef in self.coef_.T
                    ]
                ).T
        else:
            if np.isscalar(self.lambda_):
                return mo.safe_sparse_dot(X, self.coef_, backend=self.backend)
            else:
                return jnp.array(
                    [
                        mo.safe_sparse_dot(X, coef, backend=self.backend)
                        for coef in self.coef_.T
                    ]
                ).T
