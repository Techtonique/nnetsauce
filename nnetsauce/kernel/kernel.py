from scipy.special import softmax
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, RegressorMixin
import numpy as np

try:
    import jax.numpy as jnp
    from jax.scipy.special import gammaln, kv
    from jax.nn import softmax as jaxsoftmax
    from jax import jit

    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False


class KernelRidge(BaseEstimator, RegressorMixin):
    """
    Kernel Ridge Regression with optional GPU support, Matérn kernels, and automatic input standardization.

    Parameters:
    - alpha: float
        Regularization parameter.
    - kernel: str
        Kernel type ("linear", "rbf", or "matern").
    - gamma: float
        Kernel coefficient for "rbf". Ignored for other kernels.
    - nu: float
        Smoothness parameter for the Matérn kernel. Default is 1.5.
    - length_scale: float
        Length scale parameter for the Matérn kernel. Default is 1.0.
    - backend: str
        "cpu" or "gpu" (uses JAX if "gpu").
    """

    def __init__(
        self,
        alpha=1.0,
        kernel="rbf",
        gamma=None,
        nu=1.5,
        length_scale=1.0,
        backend="cpu",
    ):
        self.alpha = alpha
        self.alpha_ = alpha
        self.kernel = kernel
        self.gamma = gamma
        self.nu = nu
        self.length_scale = length_scale
        self.backend = backend
        self.scaler = StandardScaler()

        if backend == "gpu" and not JAX_AVAILABLE:
            raise ImportError(
                "JAX is not installed. Please install JAX to use the GPU backend."
            )

    def _linear_kernel(self, X, Y):
        return jnp.dot(X, Y.T) if self.backend == "gpu" else np.dot(X, Y.T)

    def _rbf_kernel(self, X, Y):
        if self.gamma is None:
            self.gamma = 1.0 / X.shape[1]
        if self.backend == "gpu":
            sq_dists = (
                jnp.sum(X**2, axis=1)[:, None]
                + jnp.sum(Y**2, axis=1)
                - 2 * jnp.dot(X, Y.T)
            )
            return jnp.exp(-self.gamma * sq_dists)
        else:
            sq_dists = (
                np.sum(X**2, axis=1)[:, None]
                + np.sum(Y**2, axis=1)
                - 2 * np.dot(X, Y.T)
            )
            return np.exp(-self.gamma * sq_dists)

    def _matern_kernel(self, X, Y):
        """
        Compute the Matérn kernel using JAX for GPU or NumPy for CPU.

        Parameters:
        - X: array-like, shape (n_samples_X, n_features)
        - Y: array-like, shape (n_samples_Y, n_features)

        Returns:
        - Kernel matrix, shape (n_samples_X, n_samples_Y)
        """
        if self.backend == "gpu":
            # Compute pairwise distances
            dists = jnp.sqrt(
                jnp.sum((X[:, None, :] - Y[None, :, :]) ** 2, axis=2)
            )
            scaled_dists = jnp.sqrt(2 * self.nu) * dists / self.length_scale

            # Matérn kernel formula
            coeff = (2 ** (1 - self.nu)) / jnp.exp(gammaln(self.nu))
            matern_kernel = (
                coeff * (scaled_dists**self.nu) * kv(self.nu, scaled_dists)
            )
            matern_kernel = jnp.where(
                dists == 0, 1.0, matern_kernel
            )  # Handle the case where distance is 0
            return matern_kernel
        else:
            # Use NumPy for CPU
            from scipy.special import (
                gammaln,
                kv,
            )  # Ensure scipy.special is used for CPU

            dists = np.sqrt(
                np.sum((X[:, None, :] - Y[None, :, :]) ** 2, axis=2)
            )
            scaled_dists = np.sqrt(2 * self.nu) * dists / self.length_scale

            # Matérn kernel formula
            coeff = (2 ** (1 - self.nu)) / np.exp(gammaln(self.nu))
            matern_kernel = (
                coeff * (scaled_dists**self.nu) * kv(self.nu, scaled_dists)
            )
            matern_kernel = np.where(
                dists == 0, 1.0, matern_kernel
            )  # Handle the case where distance is 0
            return matern_kernel

    def _get_kernel(self, X, Y):
        if self.kernel == "linear":
            return self._linear_kernel(X, Y)
        elif self.kernel == "rbf":
            return self._rbf_kernel(X, Y)
        elif self.kernel == "matern":
            return self._matern_kernel(X, Y)
        else:
            raise ValueError(f"Unsupported kernel: {self.kernel}")

    def fit(self, X, y):
        """
        Fit the Kernel Ridge Regression model.

        Parameters:
        - X: array-like, shape (n_samples, n_features)
            Training data.
        - y: array-like, shape (n_samples,)
            Target values.
        """
        # Standardize the inputs
        X = self.scaler.fit_transform(X)
        self.X_fit_ = X

        # Center the response
        self.y_mean_ = np.mean(y)
        y_centered = y - self.y_mean_

        n_samples = X.shape[0]

        # Compute the kernel matrix
        K = self._get_kernel(X, X)
        self.K_ = K
        self.y_fit_ = y_centered

        if isinstance(self.alpha, (list, np.ndarray)):
            # If alpha is a list or array, compute LOOE for each alpha
            self.alphas_ = self.alpha  # Store the list of alphas
            self.dual_coefs_ = []  # Store dual coefficients for each alpha
            self.looe_ = []  # Store LOOE for each alpha

            for alpha in self.alpha:
                G = K + alpha * np.eye(n_samples)
                G_inv = np.linalg.inv(G)
                diag_G_inv = np.diag(G_inv)
                dual_coef = np.linalg.solve(G, y_centered)
                looe = np.sum((dual_coef / diag_G_inv) ** 2)  # Compute LOOE
                self.dual_coefs_.append(dual_coef)
                self.looe_.append(looe)

            # Select the best alpha based on the smallest LOOE
            best_index = np.argmin(self.looe_)
            self.alpha_ = self.alpha[best_index]
            self.dual_coef_ = self.dual_coefs_[best_index]
        else:
            # If alpha is a single value, proceed as usual
            if self.backend == "gpu":
                self.dual_coef_ = jnp.linalg.solve(
                    K + self.alpha * jnp.eye(n_samples), y_centered
                )
            else:
                self.dual_coef_ = np.linalg.solve(
                    K + self.alpha * np.eye(n_samples), y_centered
                )

        return self

    def predict(self, X, probs=False):
        """
        Predict using the Kernel Ridge Regression model.

        Parameters:
        - X: array-like, shape (n_samples, n_features)
            Test data.

        Returns:
        - Predicted values, shape (n_samples,).
        """
        # Standardize the inputs
        X = self.scaler.transform(X)
        K = self._get_kernel(X, self.X_fit_)
        if self.backend == "gpu":
            preds = jnp.dot(K, self.dual_coef_) + self.y_mean_
            if probs:
                # Compute similarity to self.X_fit_
                similarities = jnp.dot(
                    preds, self.X_fit_.T
                )  # Shape: (n_samples, n_fit_)
                # Apply softmax to get probabilities
                return jaxsoftmax(similarities, axis=1)
            return preds
        else:
            preds = np.dot(K, self.dual_coef_) + self.y_mean_
            if probs:
                # Compute similarity to self.X_fit_
                similarities = np.dot(
                    preds, self.X_fit_.T
                )  # Shape: (n_samples, n_fit_)
                # Apply softmax to get probabilities
                return softmax(similarities, axis=1)
            return preds

    def partial_fit(self, X, y):
        """
        Incrementally fit the Kernel Ridge Regression model with new data using a recursive approach.

        Parameters:
        - X: array-like, shape (n_samples, n_features)
            New training data.
        - y: array-like, shape (n_samples,)
            New target values.

        Returns:
        - self: object
            The updated model.
        """
        # Standardize the inputs
        X = (
            self.scaler.fit_transform(X)
            if not hasattr(self, "X_fit_")
            else self.scaler.transform(X)
        )

        if not hasattr(self, "X_fit_"):
            # Initialize with the first batch of data
            self.X_fit_ = X

            # Center the response
            self.y_mean_ = np.mean(y)
            y_centered = y - self.y_mean_
            self.y_fit_ = y_centered

            n_samples = X.shape[0]

            # Compute the kernel matrix for the initial data
            self.K_ = self._get_kernel(X, X)

            # Initialize dual coefficients for each alpha
            if isinstance(self.alpha, (list, np.ndarray)):
                self.dual_coefs_ = [np.zeros(n_samples) for _ in self.alpha]
            else:
                self.dual_coef_ = np.zeros(n_samples)
        else:
            # Incrementally update with new data
            y_centered = y - self.y_mean_  # Center the new batch of responses
            for x_new, y_new in zip(X, y_centered):
                x_new = x_new.reshape(1, -1)  # Ensure x_new is 2D
                k_new = self._get_kernel(self.X_fit_, x_new).flatten()

                # Compute the kernel value for the new data point
                k_self = self._get_kernel(x_new, x_new).item()

                if isinstance(self.alpha, (list, np.ndarray)):
                    # Update dual coefficients for each alpha
                    for idx, alpha in enumerate(self.alpha):
                        gamma_new = 1 / (k_self + alpha)
                        residual = y_new - np.dot(self.dual_coefs_[idx], k_new)
                        self.dual_coefs_[idx] = np.append(
                            self.dual_coefs_[idx], gamma_new * residual
                        )
                else:
                    # Update dual coefficients for a single alpha
                    gamma_new = 1 / (k_self + self.alpha)
                    residual = y_new - np.dot(self.dual_coef_, k_new)
                    self.dual_coef_ = np.append(
                        self.dual_coef_, gamma_new * residual
                    )

                # Update the kernel matrix
                self.K_ = np.block(
                    [
                        [self.K_, k_new[:, None]],
                        [k_new[None, :], np.array([[k_self]])],
                    ]
                )

                # Update the stored data
                self.X_fit_ = np.vstack([self.X_fit_, x_new])
                self.y_fit_ = np.append(self.y_fit_, y_new)

        # Select the best alpha based on LOOE after the batch
        if isinstance(self.alpha, (list, np.ndarray)):
            self.looe_ = []
            for idx, alpha in enumerate(self.alpha):
                G = self.K_ + alpha * np.eye(self.K_.shape[0])
                G_inv = np.linalg.inv(G)
                diag_G_inv = np.diag(G_inv)
                looe = np.sum((self.dual_coefs_[idx] / diag_G_inv) ** 2)
                self.looe_.append(looe)

            # Select the best alpha
            best_index = np.argmin(self.looe_)
            self.alpha_ = self.alpha[best_index]
            self.dual_coef_ = self.dual_coefs_[best_index]

        return self
