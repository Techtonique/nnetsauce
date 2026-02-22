import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array


class ElasticNet2Regressor(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        n_hidden_features=100,
        alpha=1.0,
        l1_ratio=0.5,
        lambd=0.1,
        activation_name="tanh",
        a=0.01,
        max_iter=1000,
        tol=1e-4,
        random_state=None,
    ):
        self.n_hidden_features = n_hidden_features
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.lambd = lambd
        self.activation_name = activation_name
        self.a = a
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def _activation(self, Z):
        if self.activation_name == "relu":
            return np.maximum(0, Z)
        elif self.activation_name == "tanh":
            return np.tanh(Z)
        elif self.activation_name == "sigmoid":
            return 1 / (1 + np.exp(-Z))
        elif self.activation_name == "prelu":
            return np.where(Z > 0, Z, self.a * Z)
        elif self.activation_name == "elu":
            return np.where(Z > 0, Z, self.a * (np.exp(Z) - 1))
        else:
            raise ValueError(f"Unknown activation: {self.activation_name}")

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        rng = np.random.RandomState(self.random_state)

        # Standardize inputs
        self.X_mean_ = X.mean(axis=0)
        self.X_std_ = X.std(axis=0) + 1e-8
        X_scaled = (X - self.X_mean_) / self.X_std_

        # Center response
        self.y_mean_ = y.mean()
        y_centered = y - self.y_mean_

        # Random feature mapping
        self.W_in_ = rng.randn(X.shape[1], self.n_hidden_features)
        self.b_in_ = rng.randn(self.n_hidden_features)
        H = self._activation(X_scaled @ self.W_in_ + self.b_in_)

        # Doubly-constrained optimization with Elastic Net
        beta = np.zeros(self.n_hidden_features)

        for _ in range(self.max_iter):
            beta_old = beta.copy()

            # Gradient descent step with projection
            grad = H.T @ (H @ beta - y_centered) / len(y)
            step = 0.01 / (1 + self.alpha * (1 - self.l1_ratio))

            # Soft thresholding (L1)
            beta = beta - step * grad
            threshold = step * self.alpha * self.l1_ratio
            beta = np.sign(beta) * np.maximum(np.abs(beta) - threshold, 0)

            # L2 projection (constraint)
            norm = np.linalg.norm(beta)
            if norm > self.lambd:
                beta = beta * (self.lambd / norm)

            if np.linalg.norm(beta - beta_old) < self.tol:
                break

        self.beta_ = beta
        return self

    def predict(self, X):
        X = check_array(X)
        X_scaled = (X - self.X_mean_) / self.X_std_
        H = self._activation(X_scaled @ self.W_in_ + self.b_in_)
        return H @ self.beta_ + self.y_mean_
