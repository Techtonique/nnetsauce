import jax
import jax.numpy as jnp
import numpy as np
from scipy.stats import qmc, norm
from functools import partial
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

# Import scikit-optimize


class Ridge2Forecaster:
    """Vectorized Ridge2 RVFL for multivariate time series forecasting.

    Parameters
    ----------
    lags : int, optional
        Number of lags to use for feature engineering, by default 1
    nb_hidden : int, optional
        Number of hidden units, by default 5
    activ : str, optional
        Activation function, by default 'relu'
    lambda_1 : float, optional
        Ridge regularization parameter for input features, by default 0.1
    lambda_2 : float, optional
        Ridge regularization parameter for hidden units, by default 0.1
    nodes_sim : str, optional
        Type of quasi-random sequence for weight initialization, by default 'sobol'
    seed : int, optional
        Random seed for reproducibility, by default 42
    """

    def __init__(
        self,
        lags=1,
        nb_hidden=5,
        activ="relu",
        lambda_1=0.1,
        lambda_2=0.1,
        nodes_sim="sobol",
        seed=42,
    ):
        self.lags = lags
        self.nb_hidden = nb_hidden
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.nodes_sim = nodes_sim
        self.seed = seed
        self.coef_ = None

        # Activation functions
        activations = {
            "relu": lambda x: jnp.maximum(0, x),
            "sigmoid": lambda x: 1 / (1 + jnp.exp(-x)),
            "tanh": jnp.tanh,
            "linear": lambda x: x,
        }
        self.activation = jax.jit(activations[activ])

    def _create_lags(self, y):
        """Create lagged feature matrix (vectorized)."""
        n, p = y.shape
        X = jnp.concatenate(
            [y[self.lags - i - 1: n - i - 1] for i in range(self.lags)], axis=1
        )
        Y = y[self.lags:]
        return X, Y

    def _init_weights(self, n_features):
        """Initialize hidden layer weights using quasi-random sequences."""
        total_dim = n_features * self.nb_hidden

        if self.nodes_sim == "sobol":
            sampler = qmc.Sobol(d=total_dim, scramble=False, seed=self.seed)
            W = sampler.random(1).reshape(n_features, self.nb_hidden)
            W = 2 * W - 1
        else:
            key = jax.random.PRNGKey(self.seed)
            W = jax.random.uniform(
                key, (n_features, self.nb_hidden), minval=-1, maxval=1
            )

        return jnp.array(W)

    @partial(jax.jit, static_argnums=(0,))
    def _compute_hidden(self, X, W):
        """Compute hidden layer features (vectorized)."""
        return self.activation(X @ W)

    @partial(jax.jit, static_argnums=(0,))
    def _solve_ridge2(self, X, H, Y):
        """Solve ridge regression with dual regularization."""
        n, p_x = X.shape
        _, p_h = H.shape

        Y_mean = jnp.mean(Y, axis=0)
        Y_c = Y - Y_mean

        X_mean = jnp.mean(X, axis=0)
        X_std = jnp.std(X, axis=0)
        X_std = jnp.where(X_std == 0, 1.0, X_std)
        X_s = (X - X_mean) / X_std

        H_mean = jnp.mean(H, axis=0)
        H_std = jnp.std(H, axis=0)
        H_std = jnp.where(H_std == 0, 1.0, H_std)
        H_s = (H - H_mean) / H_std

        XX = X_s.T @ X_s + self.lambda_1 * jnp.eye(p_x)
        XH = X_s.T @ H_s
        HH = H_s.T @ H_s + self.lambda_2 * jnp.eye(p_h)

        XX_inv = jnp.linalg.inv(XX)
        S = HH - XH.T @ XX_inv @ XH
        S_inv = jnp.linalg.inv(S)

        XY = X_s.T @ Y_c
        HY = H_s.T @ Y_c

        beta = XX_inv @ (XY - XH @ S_inv @ (HY - XH.T @ XX_inv @ XY))
        gamma = S_inv @ (HY - XH.T @ beta)
        self.coef_ = jnp.concatenate([beta, gamma], axis=1)

        return beta, gamma, Y_mean, X_mean, X_std, H_mean, H_std

    def fit(self, y):
        """Fit the Ridge2 model.

        Parameters
        ----------
        y : array-like of shape (n_samples,)
            Target values.
        """
        y = jnp.array(y)
        if y.ndim == 1:
            y = y[:, None]

        X, Y = self._create_lags(y)
        self.n_series = Y.shape[1]

        self.W = self._init_weights(X.shape[1])
        H = self._compute_hidden(X, self.W)

        (
            self.beta,
            self.gamma,
            self.Y_mean,
            self.X_mean,
            self.X_std,
            self.H_mean,
            self.H_std,
        ) = self._solve_ridge2(X, H, Y)

        # Compute residuals for prediction intervals
        X_s = (X - self.X_mean) / self.X_std
        H_s = (H - self.H_mean) / self.H_std
        fitted = X_s @ self.beta + H_s @ self.gamma + self.Y_mean
        self.residuals = np.array(Y - fitted)

        self.last_obs = y[-self.lags:]
        return self

    @partial(jax.jit, static_argnums=(0,))
    def _predict_step(self, x_new):
        """Single prediction step (JIT-compiled).

        Parameters
        ----------
        x_new : array-like of shape (n_features,)
            New input data.

        Returns
        -------
        y_next : float
            Next-step prediction.
        """
        x_s = (x_new - self.X_mean) / self.X_std
        h = self.activation(x_s @ self.W)
        h_s = (h - self.H_mean) / self.H_std
        return x_s @ self.beta + h_s @ self.gamma + self.Y_mean

    def _forecast(self, h=5):
        """Generate h-step ahead recursive forecasts.

        Parameters
        ----------
        h : int, optional
            Number of steps to forecast, by default 5

        Returns
        -------
        forecasts : array-like of shape (h,)
            Forecasted values.
        """
        forecasts = []
        current = self.last_obs.copy()

        for _ in range(h):
            x_new = current.flatten()[None, :]
            y_next = self._predict_step(x_new)[0]
            forecasts.append(y_next)
            current = jnp.vstack([current[1:], y_next])

        return jnp.array(forecasts)

    def predict(self, h=5, level=None, method="gaussian", B=100):
        """Generate prediction intervals with proper uncertainty propagation.

        Parameters
        ----------
        h : int, optional
            Number of steps to forecast, by default 5
        level : float, optional
            Confidence level for prediction intervals, by default None
        method : str, optional
            Method for prediction intervals ('gaussian' or 'bootstrap'), by default 'gaussian'
        B : int, optional
            Number of bootstrap samples, by default 100

        Returns
        -------
        point_forecast : array-like of shape (h,)
            Point forecasted values.
        lower : array-like of shape (h,)
            Lower bounds of prediction intervals.
        upper : array-like of shape (h,)
            Upper bounds of prediction intervals.
        """

        point_forecast = self._forecast(h)

        if level is None:
            return point_forecast

        # probabilistic prediction intervals
        if method == "gaussian":
            # Use residual std with horizon-dependent scaling
            residual_std = np.std(self.residuals, axis=0)
            z = norm.ppf(1 - (1 - level / 100) / 2)

            # Scale uncertainty by sqrt(h) for each horizon
            horizon_scale = np.sqrt(np.arange(1, h + 1))[:, None]
            std_expanded = residual_std * horizon_scale

            lower = point_forecast - z * std_expanded
            upper = point_forecast + z * std_expanded

        elif method == "bootstrap":
            # Proper residual bootstrap
            key = jax.random.PRNGKey(self.seed)
            n_residuals = len(self.residuals)
            sims = []

            for _ in range(B):
                key, subkey = jax.random.split(key)
                boot_indices = np.random.choice(
                    n_residuals, size=h, replace=True
                )
                boot_resids = self.residuals[boot_indices]

                current = self.last_obs.copy()
                path = []

                for t in range(h):
                    x_new = current.flatten()[None, :]
                    y_pred = self._predict_step(x_new)[0]
                    y_sim = y_pred + boot_resids[t]
                    path.append(y_sim)
                    current = jnp.vstack([current[1:], y_sim])

                sims.append(jnp.array(path))

            sims = jnp.array(sims)
            lower = jnp.percentile(sims, (100 - level) / 2, axis=0)
            upper = jnp.percentile(sims, 100 - (100 - level) / 2, axis=0)

        return {
            "mean": np.array(point_forecast),
            "lower": np.array(lower),
            "upper": np.array(upper),
        }
