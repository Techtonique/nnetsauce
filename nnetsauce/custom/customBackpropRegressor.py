import jax
import jax.numpy as jnp
import nnetsauce as ns  # adjust if your import path differs
import pandas as pd
import numpy as np

from .custom import Custom
from sklearn.base import RegressorMixin
from copy import deepcopy
from collections import namedtuple
from tqdm import tqdm
from sklearn.base import BaseEstimator, RegressorMixin
from scipy.stats import norm
from sklearn.metrics import mean_pinball_loss

# import your matrix operations helper if needed (mo.rbind)


class CustomBackPropRegressor(Custom, RegressorMixin):
    """
    Finite difference trainer for nnetsauce models.

    Parameters
    ----------

    base_model : str
        The name of the base model (e.g., 'RidgeCV').

    type_grad : {'finitediff', 'autodiff'}, optional
        Type of gradient computation to use (default='finitediff').

    lr : float, optional
        Learning rate for optimization (default=1e-4).

    optimizer : {'gd', 'sgd', 'adam', 'cd'}, optional
        Optimization algorithm: gradient descent ('gd'), stochastic gradient descent ('sgd'),
        Adam ('adam'), or coordinate descent ('cd'). Default is 'gd'.

    eps : float, optional
        Scaling factor for adaptive finite difference step size (default=1e-3).

    batch_size : int, optional
        Batch size for 'sgd' optimizer (default=32).

    alpha : float, optional
        Elastic net penalty strength (default=0.0).

    l1_ratio : float, optional
        Elastic net mixing parameter (0 = Ridge, 1 = Lasso, default=0.0).

    type_loss : {'mse', 'quantile'}, optional
        Type of loss function to use (default='mse').

    q : float, optional
        Quantile for quantile loss (default=0.5).

    **kwargs
        Additional parameters to pass to the scikit-learn model.

    """

    def __init__(
        self,
        base_model,
        type_grad="finitediff",
        lr=1e-4,
        optimizer="gd",
        eps=1e-3,
        batch_size=32,
        alpha=0.0,
        l1_ratio=0.0,
        type_loss="mse",
        q=0.5,
        backend="cpu",
        **kwargs,
    ):
        super().__init__(base_model, True, **kwargs)
        self.base_model = base_model
        self.custom_kwargs = kwargs
        self.backend = backend
        self.model = ns.CustomRegressor(
            self.base_model, backend=self.backend, **self.custom_kwargs
        )
        assert isinstance(
            self.model, ns.CustomRegressor
        ), "'model' must be of class ns.CustomRegressor"
        self.type_grad = type_grad
        self.lr = lr
        self.optimizer = optimizer
        self.eps = eps
        self.loss_history_ = []
        self.opt_state = None
        self.batch_size = batch_size  # for SGD
        self.loss_history_ = []
        self._cd_index = 0  # For coordinate descent
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.type_loss = type_loss
        self.q = q

    def _loss(self, X, y, **kwargs):
        """
        Compute the loss (with elastic net penalty) for the current model.

        Parameters
        ----------

        X : array-like of shape (n_samples, n_features)
            Input data.

        y : array-like of shape (n_samples,)
            Target values.

        **kwargs
            Additional keyword arguments for loss calculation.

        Returns
        -------
        float
            The computed loss value.
        """
        y_pred = self.model.predict(X)
        if self.type_loss == "mse":
            loss = np.mean((y - y_pred) ** 2)
        elif self.type_loss == "quantile":
            loss = mean_pinball_loss(y, y_pred, alpha=self.q, **kwargs)
        W = self.model.W_
        l1 = np.sum(np.abs(W))
        l2 = np.sum(W**2)
        return loss + self.alpha * (
            self.l1_ratio * l1 + 0.5 * (1 - self.l1_ratio) * l2
        )

    def _compute_grad(self, X, y):
        """
        Compute the gradient of the loss with respect to W_ using finite differences.

        Parameters
        ----------

        X : array-like of shape (n_samples, n_features)
            Input data.

        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------

        ndarray
            Gradient array with the same shape as W_.
        """
        if self.type_grad == "autodiff":
            raise NotImplementedError(
                "Automatic differentiation is not implemented yet."
            )
            # Use JAX for automatic differentiation
            W = deepcopy(self.model.W_)
            W_flat = W.flatten()
            n_params = W_flat.size

            def loss_fn(W_flat):
                W_reshaped = W_flat.reshape(W.shape)
                self.model.W_ = W_reshaped
                return self._loss(X, y)

            grad_fn = jax.grad(loss_fn)
            grad_flat = grad_fn(W_flat)
            grad = grad_flat.reshape(W.shape)

            # Add elastic net gradient
            l1_grad = self.alpha * self.l1_ratio * np.sign(W)
            l2_grad = self.alpha * (1 - self.l1_ratio) * W
            grad += l1_grad + l2_grad

            self.model.W_ = W
            return grad

        # Finite difference gradient computation
        W = deepcopy(self.model.W_)
        shape = W.shape
        W_flat = W.flatten()
        n_params = W_flat.size

        # Adaptive finite difference step
        h_vec = self.eps * np.maximum(1.0, np.abs(W_flat))
        eye = np.eye(n_params)

        loss_plus = np.zeros(n_params)
        loss_minus = np.zeros(n_params)

        for i in range(n_params):
            h_i = h_vec[i]
            Wp = W_flat.copy()
            Wp[i] += h_i
            Wm = W_flat.copy()
            Wm[i] -= h_i

            self.model.W_ = Wp.reshape(shape)
            loss_plus[i] = self._loss(X, y)

            self.model.W_ = Wm.reshape(shape)
            loss_minus[i] = self._loss(X, y)

        grad = ((loss_plus - loss_minus) / (2 * h_vec)).reshape(shape)

        # Add elastic net gradient
        l1_grad = self.alpha * self.l1_ratio * np.sign(W)
        l2_grad = self.alpha * (1 - self.l1_ratio) * W
        grad += l1_grad + l2_grad

        self.model.W_ = W  # restore original
        return grad

    def fit(
        self,
        X,
        y,
        epochs=10,
        verbose=True,
        show_progress=True,
        sample_weight=None,
        **kwargs,
    ):
        """
        Fit the model using finite difference optimization.

        Parameters
        ----------

        X : array-like of shape (n_samples, n_features)
            Training data.

        y : array-like of shape (n_samples,)
            Target values.

        epochs : int, optional
            Number of optimization steps (default=10).

        verbose : bool, optional
            Whether to print progress messages (default=True).

        show_progress : bool, optional
            Whether to show tqdm progress bar (default=True).

        sample_weight : array-like, optional
            Sample weights.

        **kwargs
            Additional keyword arguments.

        Returns
        -------

        self : object
            Returns self.
        """

        self.model.fit(X, y)

        iterator = tqdm(range(epochs)) if show_progress else range(epochs)

        for epoch in iterator:
            grad = self._compute_grad(X, y)

            if self.optimizer == "gd":
                self.model.W_ -= self.lr * grad
                self.model.W_ = np.clip(self.model.W_, 0, 1)
                # print("self.model.W_", self.model.W_)

            elif self.optimizer == "sgd":
                # Sample a mini-batch for stochastic gradient
                n_samples = X.shape[0]
                idxs = np.random.choice(
                    n_samples, self.batch_size, replace=False
                )
                if isinstance(X, pd.DataFrame):
                    X_batch = X.iloc[idxs, :]
                else:
                    X_batch = X[idxs, :]
                y_batch = y[idxs]
                grad = self._compute_grad(X_batch, y_batch)

                self.model.W_ -= self.lr * grad
                self.model.W_ = np.clip(self.model.W_, 0, 1)

            elif self.optimizer == "adam":
                if self.opt_state is None:
                    self.opt_state = {
                        "m": np.zeros_like(grad),
                        "v": np.zeros_like(grad),
                        "t": 0,
                    }
                beta1, beta2, eps = 0.9, 0.999, 1e-8
                self.opt_state["t"] += 1
                self.opt_state["m"] = (
                    beta1 * self.opt_state["m"] + (1 - beta1) * grad
                )
                self.opt_state["v"] = beta2 * self.opt_state["v"] + (
                    1 - beta2
                ) * (grad**2)
                m_hat = self.opt_state["m"] / (1 - beta1 ** self.opt_state["t"])
                v_hat = self.opt_state["v"] / (1 - beta2 ** self.opt_state["t"])

                self.model.W_ -= self.lr * m_hat / (np.sqrt(v_hat) + eps)
                self.model.W_ = np.clip(self.model.W_, 0, 1)
                # print("self.model.W_", self.model.W_)

            elif self.optimizer == "cd":  # coordinate descent
                W_shape = self.model.W_.shape
                W_flat_size = self.model.W_.size
                W_flat = self.model.W_.flatten()
                grad_flat = grad.flatten()

                # Update only one coordinate per epoch (cyclic)
                idx = self._cd_index % W_flat_size
                W_flat[idx] -= self.lr * grad_flat[idx]
                # Clip the updated value
                W_flat[idx] = np.clip(W_flat[idx], 0, 1)

                # Restore W_
                self.model.W_ = W_flat.reshape(W_shape)

                self._cd_index += 1

            else:
                raise ValueError(f"Unsupported optimizer: {self.optimizer}")

            loss = self._loss(X, y)
            self.loss_history_.append(loss)

            if verbose:
                print(f"Epoch {epoch+1}: Loss = {loss:.6f}")

        # if sample_weights, else: (must use self.row_index)
        if sample_weight in kwargs:
            self.model.fit(
                X,
                y,
                sample_weight=sample_weight[self.index_row_].ravel(),
                **kwargs,
            )

            return self

        return self

    def predict(self, X, level=95, method="splitconformal", **kwargs):
        """
        Predict using the trained model.

        Parameters
        ----------

        X : array-like of shape (n_samples, n_features)
            Input data.

        level : int, optional
            Level of confidence for prediction intervals (default=95).

        method : {'splitconformal', 'localconformal'}, optional
            Method for conformal prediction (default='splitconformal').

        **kwargs
            Additional keyword arguments. Use `return_pi=True` for prediction intervals,
            or `return_std=True` for standard deviation estimates.

        Returns
        -------

        array or tuple
            Model predictions, or a tuple with prediction intervals or standard deviations if requested.
        """
        if "return_std" in kwargs:
            alpha = 100 - level
            pi_multiplier = norm.ppf(1 - alpha / 200)

            if len(X.shape) == 1:
                n_features = X.shape[0]
                new_X = mo.rbind(
                    X.reshape(1, n_features),
                    np.ones(n_features).reshape(1, n_features),
                )

                mean_, std_ = self.model.predict(new_X, return_std=True)[0]

                preds = mean_
                lower = mean_ - pi_multiplier * std_
                upper = mean_ + pi_multiplier * std_

                DescribeResults = namedtuple(
                    "DescribeResults", ["mean", "std", "lower", "upper"]
                )

                return DescribeResults(preds, std_, lower, upper)

            # len(X.shape) > 1
            mean_, std_ = self.model.predict(X, return_std=True)

            preds = mean_
            lower = mean_ - pi_multiplier * std_
            upper = mean_ + pi_multiplier * std_

            DescribeResults = namedtuple(
                "DescribeResults", ["mean", "std", "lower", "upper"]
            )

            return DescribeResults(preds, std_, lower, upper)

        if "return_pi" in kwargs:
            assert method in (
                "splitconformal",
                "localconformal",
            ), "method must be in ('splitconformal', 'localconformal')"
            self.pi = ns.PredictionInterval(
                obj=self,
                method=method,
                level=level,
                type_pi=self.type_pi,
                replications=self.replications,
                kernel=self.kernel,
            )

            if len(self.X_.shape) == 1:
                if isinstance(X, pd.DataFrame):
                    self.X_ = pd.DataFrame(
                        self.X_.values.reshape(1, -1), columns=self.X_.columns
                    )
                else:
                    self.X_ = self.X_.reshape(1, -1)
                self.y_ = np.array([self.y_])

            self.pi.fit(self.X_, self.y_)
            # self.X_ = None # consumes memory to keep, dangerous to delete (side effect)
            # self.y_ = None # consumes memory to keep, dangerous to delete (side effect)
            preds = self.pi.predict(X, return_pi=True)
            return preds

        # "return_std" not in kwargs
        if len(X.shape) == 1:
            n_features = X.shape[0]
            new_X = mo.rbind(
                X.reshape(1, n_features),
                np.ones(n_features).reshape(1, n_features),
            )

            return (0 + self.model.predict(new_X, **kwargs))[0]

        # len(X.shape) > 1
        return self.model.predict(X, **kwargs)
