# Authors: Thierry Moudiki
#
# License: BSD 3 Clear

import numpy as np
import platform
from scipy.optimize import minimize
import sklearn.metrics as skm
from .ridge2 import Ridge2
from ..utils import matrixops as mo
from ..utils import misc as mx
from sklearn.base import RegressorMixin
from scipy.special import logsumexp
from scipy.linalg import pinv

try:
    import jax
    import jax.numpy as jnp
    from jax.numpy.linalg import pinv as jpinv

    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    jnp = np


class Ridge2MultiOutputRegressor(Ridge2, RegressorMixin):
    """Ridge regression with 2 regularization parameters for multiple outputs (zero-loop, JAX-optimized)

    Parameters:

        n_hidden_features: int
            number of nodes in the hidden layer

        activation_name: str
            activation function: 'relu', 'tanh', 'sigmoid', 'prelu' or 'elu'

        a: float
            hyperparameter for 'prelu' or 'elu' activation function

        nodes_sim: str
            type of simulation for the nodes: 'sobol', 'hammersley', 'halton',
            'uniform'

        bias: boolean
            indicates if the hidden layer contains a bias term (True) or not
            (False)

        dropout: float
            regularization parameter; (random) percentage of nodes dropped out
            of the training

        n_clusters: int
            number of clusters for 'kmeans' or 'gmm' clustering (could be 0:
                no clustering)

        cluster_encode: bool
            defines how the variable containing clusters is treated (default is one-hot)
            if `False`, then labels are used, without one-hot encoding

        type_clust: str
            type of clustering method: currently k-means ('kmeans') or Gaussian
            Mixture Model ('gmm')

        type_scaling: a tuple of 3 strings
            scaling methods for inputs, hidden layer, and clustering respectively
            (and when relevant).
            Currently available: standardization ('std') or MinMax scaling ('minmax')

        lambda1: float
            regularization parameter on direct link

        lambda2: float
            regularization parameter on hidden layer

        seed: int
            reproducibility seed for nodes_sim=='uniform'

        backend: str
            'cpu' or 'gpu' or 'tpu'

    Attributes:

        beta_: {array-like}, shape = [n_features, n_outputs]
            regression coefficients

        coef_: {array-like}
            alias for `beta_`, regression coefficients

        y_mean_: array-like, shape = [n_outputs]
            average response for each output

    """

    def __init__(
        self,
        n_hidden_features=5,
        activation_name="relu",
        a=0.01,
        nodes_sim="sobol",
        bias=True,
        dropout=0,
        n_clusters=2,
        cluster_encode=True,
        type_clust="kmeans",
        type_scaling=("std", "std", "std"),
        lambda1=0.1,
        lambda2=0.1,
        seed=123,
        backend="cpu",
    ):
        if not JAX_AVAILABLE and backend != "cpu":
            raise RuntimeError(
                "JAX is required for this feature. Install with: pip install yourpackage[jax]"
            )

        super().__init__(
            n_hidden_features=n_hidden_features,
            activation_name=activation_name,
            a=a,
            nodes_sim=nodes_sim,
            bias=bias,
            dropout=dropout,
            n_clusters=n_clusters,
            cluster_encode=cluster_encode,
            type_clust=type_clust,
            type_scaling=type_scaling,
            lambda1=lambda1,
            lambda2=lambda2,
            seed=seed,
            backend=backend,
        )

        self.type_fit = "regression"
        self.coef_ = None
        self.use_jax = JAX_AVAILABLE and backend in ("gpu", "tpu")

    def fit(self, X, y, **kwargs):
        """Fit Ridge model to training data (X, y) with multiple outputs.

        Args:

            X: {array-like}, shape = [n_samples, n_features]
                Training vectors, where n_samples is the number
                of samples and n_features is the number of features.

            y: array-like, shape = [n_samples] or [n_samples, n_outputs]
                Target values. Can be 1D for single output or 2D for multiple outputs.

            **kwargs: additional parameters to be passed to
                    self.cook_training_set or self.obj.fit

        Returns:

            self: object

        """

        sys_platform = platform.system()

        # Ensure y is 2D
        y = np.atleast_2d(y)
        if y.shape[0] == 1 and y.shape[1] > 1:
            y = y.T

        centered_y, scaled_Z = self.cook_training_set(y=y, X=X, **kwargs)

        n_X, p_X = X.shape
        n_Z, p_Z = scaled_Z.shape
        n_outputs = centered_y.shape[1] if centered_y.ndim > 1 else 1

        if self.n_clusters > 0:
            if self.encode_clusters == True:
                n_features = p_X + self.n_clusters
            else:
                n_features = p_X + 1
        else:
            n_features = p_X

        X_ = scaled_Z[:, 0:n_features]
        Phi_X_ = scaled_Z[:, n_features:p_Z]

        # Use JAX if available and requested
        if self.use_jax:
            X_ = jnp.array(X_)
            Phi_X_ = jnp.array(Phi_X_)
            centered_y = jnp.array(centered_y)

            # Compute all matrix operations with JAX
            B = jnp.dot(X_.T, X_) + self.lambda1 * jnp.eye(n_features)
            C = jnp.dot(Phi_X_.T, X_)
            D = jnp.dot(Phi_X_.T, Phi_X_) + self.lambda2 * jnp.eye(
                Phi_X_.shape[1]
            )

            B_inv = jpinv(B)
            W = jnp.dot(C, B_inv)
            S_mat = D - jnp.dot(W, C.T)
            S_inv = jpinv(S_mat)
            Y = jnp.dot(S_inv, W)

            # Build inverse matrix
            inv_upper = jnp.hstack([B_inv + jnp.dot(W.T, Y), -Y.T])
            inv_lower = jnp.hstack([-Y, S_inv])
            inv = jnp.vstack([inv_upper, inv_lower])

            # Compute beta for all outputs at once (vectorized)
            Z_T_y = jnp.dot(scaled_Z.T, centered_y)
            self.beta_ = jnp.dot(inv, Z_T_y)

            # Convert back to numpy
            self.beta_ = np.array(self.beta_)
        else:
            # NumPy version
            B = mo.crossprod(
                x=X_, backend=self.backend
            ) + self.lambda1 * np.diag(np.repeat(1, n_features))
            C = mo.crossprod(x=Phi_X_, y=X_, backend=self.backend)
            D = mo.crossprod(
                x=Phi_X_, backend=self.backend
            ) + self.lambda2 * np.diag(np.repeat(1, Phi_X_.shape[1]))

            if sys_platform in ("Linux", "Darwin"):
                B_inv = pinv(B) if self.backend == "cpu" else jpinv(B)
            else:
                B_inv = pinv(B)

            W = mo.safe_sparse_dot(a=C, b=B_inv, backend=self.backend)
            S_mat = D - mo.tcrossprod(x=W, y=C, backend=self.backend)

            if sys_platform in ("Linux", "Darwin"):
                S_inv = pinv(S_mat) if self.backend == "cpu" else jpinv(S_mat)
            else:
                S_inv = pinv(S_mat)

            Y = mo.safe_sparse_dot(a=S_inv, b=W, backend=self.backend)
            inv = mo.rbind(
                mo.cbind(
                    x=B_inv + mo.crossprod(x=W, y=Y, backend=self.backend),
                    y=-np.transpose(Y),
                    backend=self.backend,
                ),
                mo.cbind(x=-Y, y=S_inv, backend=self.backend),
                backend=self.backend,
            )

            # Vectorized multi-output computation (no loop)
            Z_T_y = mo.crossprod(x=scaled_Z, y=centered_y, backend=self.backend)
            self.beta_ = mo.safe_sparse_dot(
                a=inv, b=Z_T_y, backend=self.backend
            )

        self.coef_ = self.beta_  # sklearn compatibility

        return self

    def predict(self, X, **kwargs):
        """Predict test data X for all outputs.

        Args:

            X: {array-like}, shape = [n_samples, n_features]
                Training vectors, where n_samples is the number
                of samples and n_features is the number of features.

            **kwargs: additional parameters to be passed to
                    self.cook_test_set

        Returns:

            model predictions: {array-like}, shape = [n_samples, n_outputs]

        """

        if len(X.shape) == 1:
            n_features = X.shape[0]
            new_X = mo.rbind(
                x=X.reshape(1, n_features),
                y=np.ones(n_features).reshape(1, n_features),
                backend=self.backend,
            )

            cooked = self.cook_test_set(new_X, **kwargs)

            if self.use_jax:
                cooked = jnp.array(cooked)
                predictions = self.y_mean_ + jnp.dot(cooked, self.beta_)
                return np.array(predictions[0])
            else:
                return (
                    self.y_mean_
                    + mo.safe_sparse_dot(
                        a=cooked,
                        b=self.beta_,
                        backend=self.backend,
                    )
                )[0]

        cooked = self.cook_test_set(X, **kwargs)

        if self.use_jax:
            cooked = jnp.array(cooked)
            predictions = self.y_mean_ + jnp.dot(cooked, self.beta_)
            return np.array(predictions)
        else:
            return self.y_mean_ + mo.safe_sparse_dot(
                a=cooked,
                b=self.beta_,
                backend=self.backend,
            )

    def partial_fit(self, X, y, learning_rate=0.01, decay=0.001, **kwargs):
        """Incrementally fit the Ridge model using vectorized SGD updates (zero-loop with JAX).

        Uses vectorized update rule for all outputs simultaneously.

        Args:
            X: {array-like}, shape = [n_samples, n_features]
                Training vectors for this batch

            y: array-like, shape = [n_samples] or [n_samples, n_outputs]
                Target values for this batch

            learning_rate: float, default=0.01
                Initial learning rate for SGD updates

            decay: float, default=0.001
                Learning rate decay parameter

            **kwargs: additional parameters to be passed to self.cook_training_set

        Returns:
            self: object
        """

        # Input validation
        X = np.asarray(X)
        y = np.atleast_2d(y)
        if y.shape[0] == 1 and y.shape[1] > 1:
            y = y.T

        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples")

        # Handle first call
        if not self._is_fitted:
            self.initial_learning_rate = learning_rate
            self.decay = decay
            self._step_count = 0
            self._is_fitted = True

        # Process the batch
        centered_y, scaled_Z = self.cook_training_set(y=y, X=X, **kwargs)

        # Get dimensions
        n_samples, n_features_total = scaled_Z.shape
        n_original_features = X.shape[1]
        n_outputs = centered_y.shape[1] if centered_y.ndim > 1 else 1

        # Determine feature dimensions for regularization
        if self.n_clusters > 0:
            if self.cluster_encode:
                n_direct_features = n_original_features + self.n_clusters
            else:
                n_direct_features = n_original_features + 1
        else:
            n_direct_features = n_original_features

        # Initialize beta_ if first time
        if not hasattr(self, "beta_") or self.beta_ is None:
            self.beta_ = np.zeros((n_features_total, n_outputs))

        # Create regularization mask
        reg_mask = np.concatenate(
            [
                np.full(n_direct_features, self.lambda1),
                np.full(n_features_total - n_direct_features, self.lambda2),
            ]
        )[
            :, np.newaxis
        ]  # Shape: [n_features_total, 1]

        if self.use_jax:
            # JAX vectorized implementation (fully zero-loop)
            scaled_Z = jnp.array(scaled_Z)
            centered_y = jnp.array(centered_y)
            self.beta_ = jnp.array(self.beta_)
            reg_mask = jnp.array(reg_mask)

            # Vectorized over all samples using scan
            def update_step(beta, inputs):
                step, x_i, y_i = inputs

                # Learning rate with decay
                lr = self.initial_learning_rate / (1 + self.decay * step)

                # Prediction: x_i @ beta -> [n_outputs]
                prediction = jnp.dot(x_i, beta)

                # Error: y_i - prediction -> [n_outputs]
                error = y_i - prediction

                # Gradient update (vectorized): lr * outer(x_i, error)
                gradient_update = lr * jnp.outer(x_i, error)

                # Regularization: lr * (reg_mask * beta)
                reg_update = lr * (reg_mask * beta)

                # Update: beta = beta + gradient - regularization
                beta_new = beta + gradient_update - reg_update

                return beta_new, None

            # Create step indices
            steps = jnp.arange(
                self._step_count + 1, self._step_count + n_samples + 1
            )

            # Run scan (zero-loop)
            self.beta_, _ = jax.lax.scan(
                update_step, self.beta_, (steps, scaled_Z, centered_y)
            )

            self.beta_ = np.array(self.beta_)
            self._step_count += n_samples
        else:
            # NumPy vectorized implementation (single loop over samples)
            for i in range(n_samples):
                self._step_count += 1

                # Current learning rate with decay
                current_lr = self.initial_learning_rate / (
                    1 + self.decay * self._step_count
                )

                # Current sample and target
                x_i = scaled_Z[i, :]  # [n_features_total]
                y_i = centered_y[i, :]  # [n_outputs]

                # Prediction: x_i @ beta -> [n_outputs]
                prediction = x_i @ self.beta_

                # Error: y_i - prediction -> [n_outputs]
                error = y_i - prediction

                # Vectorized gradient update: outer product
                # Shape: [n_features_total, n_outputs]
                gradient_update = current_lr * np.outer(x_i, error)

                # Vectorized regularization update
                reg_update = current_lr * (reg_mask * self.beta_)

                # Combined update
                self.beta_ += gradient_update - reg_update

        self.coef_ = self.beta_  # sklearn compatibility

        return self
