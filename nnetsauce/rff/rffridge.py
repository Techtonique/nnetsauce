import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize_scalar
from typing import Optional, Tuple, Union, Dict
from jax import random, vmap
from functools import partial
from sklearn.base import BaseEstimator, RegressorMixin


class RandomFourierFeaturesRidge(BaseEstimator, RegressorMixin):
    """
    Random Fourier Features with Bayesian Ridge Regression.

    Implements both standard (MLE) and Bayesian versions with uncertainty quantification.
    Uses data augmentation for L2 regularization via jnp.lstsq.
    """

    def __init__(
        self,
        n_features: int = 100,
        gamma: float = 1.0,
        alpha: float = 1e-6,
        include_bias: bool = True,
        random_seed: int = 42,
    ):
        """
        Parameters:
        -----------
        n_features : int
            Number of random Fourier features (D)
        gamma : float
            RBF kernel parameter: k(x,y) = exp(-gamma * ||x-y||²)
        alpha : float
            Prior precision (inverse variance) for Bayesian version
            Equivalent to regularization strength: lambda = alpha / beta
        include_bias : bool
            Whether to include a bias term
        random_seed : int
            Random seed for reproducibility
        """
        self.n_features = n_features
        self.gamma = gamma
        self.alpha = alpha
        self.include_bias = include_bias
        self.key = random.PRNGKey(random_seed)
        self.is_fitted = False

        # Bayesian parameters
        self.beta = None  # Noise precision (will be estimated from data)
        self.w_mean = None  # Posterior mean of weights
        self.w_cov = None  # Posterior covariance of weights
        self.S_N = None  # Posterior precision matrix

    def _compute_random_features(
        self, X: jnp.ndarray, W: jnp.ndarray, b: jnp.ndarray
    ) -> jnp.ndarray:
        """Compute random Fourier features: sqrt(2/D) * cos(XW + b)"""
        projection = jnp.dot(X, W) + b  # Shape: (n_samples, n_features)
        features = jnp.sqrt(2.0 / self.n_features) * jnp.cos(projection)

        if self.include_bias:
            features = jnp.concatenate(
                [jnp.ones((X.shape[0], 1)), features], axis=1
            )

        return features

    def _init_random_weights(
        self, input_dim: int
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Initialize random weights and biases for RFF"""
        # Sample from Gaussian distribution for RBF kernel
        # Variance = 2 * gamma for RBF kernel
        self.key, subkey = random.split(self.key)
        W = random.normal(
            subkey, shape=(input_dim, self.n_features)
        ) * jnp.sqrt(2.0 * self.gamma)

        self.key, subkey = random.split(self.key)
        b = random.uniform(
            subkey, shape=(1, self.n_features), minval=0, maxval=2 * jnp.pi
        )

        return W, b

    def fit(
        self,
        X: Union[jnp.ndarray, np.ndarray],
        y: Union[jnp.ndarray, np.ndarray],
        method: str = "bayesian",
        noise_variance: Optional[float] = None,
    ) -> "RandomFourierFeaturesRidge":
        """
        Fit the model using either standard or Bayesian ridge regression.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data
        y : array-like, shape (n_samples,) or (n_samples, n_targets)
            Target values
        method : str, either "standard" or "bayesian"
            "standard": Maximum likelihood estimation with L2 regularization
            "bayesian": Full Bayesian inference with uncertainty quantification
        noise_variance : float, optional
            If provided, fixes the noise variance instead of estimating it
        """
        # Convert to JAX arrays if needed
        X = jnp.asarray(X)
        y = jnp.asarray(y)

        if len(y.shape) == 1:
            y = y.reshape(-1, 1)

        n_samples, input_dim = X.shape

        # Initialize random Fourier weights
        self.W, self.b = self._init_random_weights(input_dim)

        # Compute random Fourier features
        Phi = self._compute_random_features(X, self.W, self.b)
        n_basis = Phi.shape[1]  # D + 1 if bias included

        # Store feature matrix and target values for Bayesian updates/likelihood computation
        self.Phi_train = Phi
        self.y_train = y  # Store y_train

        if method == "standard":
            # Standard ridge regression using data augmentation for regularization
            self._fit_standard(Phi, y)
        elif method == "bayesian":
            # Bayesian ridge regression
            self._fit_bayesian(Phi, y, noise_variance)
        else:
            raise ValueError("method must be 'standard' or 'bayesian'")

        self.is_fitted = True
        self.method = method
        self.input_dim = input_dim

        return self

    def _fit_standard(self, Phi: jnp.ndarray, y: jnp.ndarray) -> None:
        """Standard ridge regression using lstsq with data augmentation"""
        n_samples, n_basis = Phi.shape

        # Create augmented data for L2 regularization
        # This is equivalent to adding sqrt(alpha) * I to the design matrix
        sqrt_alpha = jnp.sqrt(self.alpha)
        Phi_aug = jnp.vstack([Phi, sqrt_alpha * jnp.eye(n_basis)])
        y_aug = jnp.vstack([y, jnp.zeros((n_basis, y.shape[1]))])

        # Solve using least squares
        # Note: jnp.linalg.lstsq is more stable than explicit normal equations
        weights, residuals, rank, s = jnp.linalg.lstsq(
            Phi_aug, y_aug, rcond=None
        )

        self.w_mean = weights
        self.weights = weights  # For compatibility

        # Estimate noise variance from residuals
        residuals = y - Phi @ weights
        self.beta = 1.0 / jnp.maximum(jnp.var(residuals), 1e-8)

    def _fit_bayesian(
        self,
        Phi: jnp.ndarray,
        y: jnp.ndarray,
        noise_variance: Optional[float] = None,
    ) -> None:
        """Bayesian ridge regression with evidence approximation"""
        n_samples, n_basis = Phi.shape

        # Initialize precision parameters
        if noise_variance is not None:
            self.beta = 1.0 / noise_variance
        else:
            # Initial estimate of beta from data
            self.beta = 1.0 / jnp.maximum(jnp.var(y), 1e-8)

        # Posterior precision matrix: S_N⁻¹ = alpha * I + beta * ΦᵀΦ
        I = jnp.eye(n_basis)
        PhiT_Phi = Phi.T @ Phi

        # Initialize with prior
        S_N_inv = self.alpha * I

        # Evidence approximation to optimize alpha, beta
        for _ in range(10):  # Iterate to converge on alpha, beta
            # Update posterior mean and covariance
            S_N = jnp.linalg.inv(S_N_inv + self.beta * PhiT_Phi)
            self.w_mean = self.beta * S_N @ Phi.T @ y

            # Update gamma (effective number of parameters)
            eigenvalues = jnp.linalg.eigvalsh(PhiT_Phi)
            gamma_val = jnp.sum(eigenvalues / (self.alpha + eigenvalues))

            # Update alpha and beta (MacKay's fixed point updates)
            if self.alpha > 0:
                self.alpha = gamma_val / jnp.sum(self.w_mean**2)

            if noise_variance is None:
                residuals = y - Phi @ self.w_mean
                self.beta = (n_samples - gamma_val) / jnp.sum(residuals**2)

            # Update precision matrix
            S_N_inv = self.alpha * I

        # Store final covariance
        self.S_N = jnp.linalg.inv(self.alpha * I + self.beta * PhiT_Phi)
        self.w_cov = self.S_N

        # Also store for compatibility
        self.weights = self.w_mean

    def transform(self, X: Union[jnp.ndarray, np.ndarray]) -> jnp.ndarray:
        """Transform input data to random Fourier feature space"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before transforming")

        X = jnp.asarray(X)
        return self._compute_random_features(X, self.W, self.b)

    def predict(
        self,
        X: Union[jnp.ndarray, np.ndarray],
        return_std: bool = False,
        return_cov: bool = False,
    ) -> Union[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]:
        """
        Make predictions, optionally with uncertainty quantification.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Input data
        return_std : bool
            If True, return standard deviation of predictive distribution
        return_cov : bool
            If True, return full covariance matrix of predictive distribution

        Returns:
        --------
        y_pred : jnp.ndarray
            Predictive mean
        y_std or y_cov : jnp.ndarray, optional
            Predictive standard deviation or covariance
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        X = jnp.asarray(X)
        Phi = self.transform(X)

        # Predictive mean
        y_pred = Phi @ self.w_mean

        if not return_std and not return_cov:
            return y_pred

        if self.method != "bayesian":
            raise ValueError(
                "Uncertainty quantification only available for Bayesian method"
            )

        # Predictive variance
        if return_cov:
            # Full predictive covariance
            # Σ_pred = (1/β) * I + Φ @ S_N @ Φᵀ
            pred_cov = (1.0 / self.beta) * jnp.eye(
                Phi.shape[0]
            ) + Phi @ self.S_N @ Phi.T
            return y_pred, pred_cov
        else:
            # Diagonal of predictive covariance (standard deviations)
            # σ²_pred = (1/β) + diag(Φ @ S_N @ Φᵀ)
            var_diag = (1.0 / self.beta) + jnp.sum(
                (Phi @ self.S_N) * Phi, axis=1
            )
            y_std = jnp.sqrt(jnp.maximum(var_diag, 0.0)).reshape(-1, 1)
            return y_pred, y_std

    def sample_posterior(
        self,
        X: Union[jnp.ndarray, np.ndarray],
        n_samples: int = 1,
        key: Optional[jax.random.PRNGKey] = None,
    ) -> jnp.ndarray:
        """
        Sample from the posterior predictive distribution.

        Parameters:
        -----------
        X : array-like
            Input data
        n_samples : int
            Number of samples to draw
        key : PRNGKey, optional
            Random key for sampling

        Returns:
        --------
        samples : jnp.ndarray, shape (n_samples, n_test_samples)
            Samples from posterior predictive distribution
        """
        if self.method != "bayesian":
            raise ValueError("Sampling only available for Bayesian method")

        if key is None:
            key = self.key

        X = jnp.asarray(X)
        Phi = self.transform(X)
        n_test = Phi.shape[0]

        # Sample weights from posterior
        key, subkey = random.split(key)
        w_samples = random.multivariate_normal(
            subkey, self.w_mean.flatten(), self.S_N, shape=(n_samples,)
        )

        # Generate predictions for each weight sample
        samples = []
        for i in range(n_samples):
            w_sample = w_samples[i].reshape(-1, 1)
            # Add noise variance
            key, subkey1, subkey2 = random.split(key, 3)
            pred_mean = Phi @ w_sample
            noise = random.normal(subkey2, shape=pred_mean.shape) / jnp.sqrt(
                self.beta
            )
            samples.append(pred_mean + noise)

        return jnp.stack(samples, axis=0)

    def log_marginal_likelihood(self) -> float:
        """
        Compute log marginal likelihood (evidence) for Bayesian model.

        Returns:
        --------
        log_evidence : float
            Log marginal likelihood p(y|X,α,β)
        """
        if self.method != "bayesian":
            raise ValueError(
                "Log marginal likelihood only available for Bayesian method"
            )

        n_samples = self.Phi_train.shape[0]
        n_basis = self.Phi_train.shape[1]

        # Log determinant term
        I = jnp.eye(n_basis)
        A = self.alpha * I + self.beta * self.Phi_train.T @ self.Phi_train
        sign, logdet_A = jnp.linalg.slogdet(A)
        logdet_term = 0.5 * (n_basis * jnp.log(self.alpha) - logdet_A)

        # Data fit term
        residuals = self.y_train - self.Phi_train @ self.w_mean
        data_fit_term = -0.5 * self.beta * jnp.sum(residuals**2)

        # Constant term
        const_term = 0.5 * n_samples * jnp.log(self.beta / (2 * jnp.pi))

        return float(logdet_term + data_fit_term + const_term)

    def get_params(self) -> Dict:
        """Get model parameters"""
        return {
            "n_features": self.n_features,
            "gamma": self.gamma,
            "alpha": self.alpha,
            "beta": self.beta if self.beta is not None else None,
            "method": self.method if hasattr(self, "method") else None,
            "input_dim": self.input_dim if hasattr(self, "input_dim") else None,
        }

    def set_params(self, **params) -> "RandomFourierFeaturesRidge":
        """Set model parameters"""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self


class RandomFourierFeaturesRidgeGCV(RandomFourierFeaturesRidge):
    """
    Extends RandomFourierFeaturesRidge with GCV for automatic
    regularization parameter selection.
    """

    def __init__(
        self,
        n_features: int = 100,
        gamma: float = 1.0,
        alpha: Optional[float] = None,
        include_bias: bool = True,
        random_seed: int = 42,
    ):
        super().__init__(n_features, gamma, alpha, include_bias, random_seed)
        self.alpha_opt = None  # Stores the GCV-optimized alpha
        self.gcv_score = None  # Stores the optimal GCV score

    def _compute_gcv(
        self,
        alpha: float,
        s_sq: jnp.ndarray,
        U: jnp.ndarray,
        y: jnp.ndarray,
        n_samples: int,
    ) -> float:
        """
        Compute GCV score for a given alpha.

        Parameters:
        -----------
        alpha : float
            Regularization parameter
        s_sq : jnp.ndarray
            Squared singular values of design matrix Φ
        U : jnp.ndarray
            Left singular vectors of Φ
        y : jnp.ndarray
            Target values
        n_samples : int
            Number of data points

        Returns:
        --------
        gcv : float
            GCV score for this alpha
        """
        # Degrees of freedom: df(α) = Σ(σ_j²/(σ_j² + α))
        df = jnp.sum(s_sq / (s_sq + alpha))

        # Compute residual sum of squares efficiently using SVD
        # y_pred = U @ (S²/(S² + α)) @ (U.T @ y)
        Uty = U.T @ y
        shrinkage = s_sq / (s_sq + alpha)
        y_pred = U @ (shrinkage * Uty)
        residuals = y - y_pred
        rss = jnp.sum(residuals**2)

        # GCV formula
        denom = (1.0 - df / n_samples) ** 2
        gcv = (rss / n_samples) / denom

        return float(gcv)

    def fit_gcv(
        self,
        X: Union[jnp.ndarray, np.ndarray],
        y: Union[jnp.ndarray, np.ndarray],
        alpha_range: Tuple[float, float] = (1e-8, 1e4),
        n_alphas: int = 50,
        method: str = "standard",
        optimize: bool = True,
    ) -> "RandomFourierFeaturesRidgeGCV":
        """
        Fit model with GCV-optimized regularization parameter.

        Parameters:
        -----------
        X : array-like
            Training data
        y : array-like
            Target values
        alpha_range : tuple
            (min_alpha, max_alpha) range to search
        n_alphas : int
            Number of alpha values to try in initial grid search
        method : str
            "standard" or "bayesian"
        optimize : bool
            If True, perform fine optimization after grid search

        Returns:
        --------
        self : fitted model
        """
        # Convert to JAX arrays
        X = jnp.asarray(X)
        y = jnp.asarray(y)

        if len(y.shape) == 1:
            y = y.reshape(-1, 1)

        n_samples, input_dim = X.shape

        # Initialize random Fourier weights
        self.W, self.b = self._init_random_weights(input_dim)

        # Compute random Fourier features
        Phi = self._compute_random_features(X, self.W, self.b)

        # Compute SVD of design matrix for efficient GCV computation
        # Φ = U @ diag(S) @ V.T
        U, S, Vt = jnp.linalg.svd(Phi, full_matrices=False)
        s_sq = S**2  # Squared singular values

        # Grid search on log scale
        alphas_grid = jnp.logspace(
            jnp.log10(alpha_range[0]), jnp.log10(alpha_range[1]), n_alphas
        )

        gcv_scores = []
        for alpha in alphas_grid:
            score = self._compute_gcv(float(alpha), s_sq, U, y, n_samples)
            gcv_scores.append(score)

        # Find best alpha from grid
        best_idx = jnp.argmin(jnp.array(gcv_scores))
        alpha_grid_opt = float(alphas_grid[best_idx])

        # Fine optimization using Brent's method
        if optimize:
            # Define objective for scipy optimizer
            def gcv_objective(log_alpha):
                alpha = 10**log_alpha
                return self._compute_gcv(alpha, s_sq, U, y, n_samples)

            # Optimize in log space
            result = minimize_scalar(
                gcv_objective,
                bounds=(jnp.log10(alpha_range[0]), jnp.log10(alpha_range[1])),
                method="bounded",
                options={"xatol": 0.1},  # Tolerance in log10 space
            )

            if result.success:
                alpha_opt = 10**result.x
                gcv_opt = result.fun
            else:
                alpha_opt = alpha_grid_opt
                gcv_opt = gcv_scores[best_idx]
        else:
            alpha_opt = alpha_grid_opt
            gcv_opt = gcv_scores[best_idx]

        # Store optimized parameters
        self.alpha_opt = alpha_opt
        self.gcv_score = gcv_opt
        self.alpha = alpha_opt  # Set as the model's alpha

        # Fit final model with optimized alpha
        if method == "standard":
            self._fit_standard(Phi, y)
        elif method == "bayesian":
            # For Bayesian version, we can use alpha as prior precision
            # Optionally optimize beta too
            self._fit_bayesian(Phi, y)
        else:
            raise ValueError("method must be 'standard' or 'bayesian'")

        self.is_fitted = True
        self.method = method
        self.input_dim = input_dim

        return self

    def fit_gcv_with_path(
        self,
        X: Union[jnp.ndarray, np.ndarray],
        y: Union[jnp.ndarray, np.ndarray],
        alpha_range: Tuple[float, float] = (1e-8, 1e4),
        n_alphas: int = 100,
        method: str = "standard",
    ) -> dict:
        """
        Fit with GCV and return full regularization path.

        Returns:
        --------
        path_info : dict
            Dictionary with alpha values, GCV scores, and metrics
        """
        X = jnp.asarray(X)
        y = jnp.asarray(y)

        if len(y.shape) == 1:
            y = y.reshape(-1, 1)

        n_samples, input_dim = X.shape

        # Initialize random features
        self.W, self.b = self._init_random_weights(input_dim)
        Phi = self._compute_random_features(X, self.W, self.b)

        # Compute SVD
        U, S, Vt = jnp.linalg.svd(Phi, full_matrices=False)
        s_sq = S**2

        # Compute GCV path
        alphas = jnp.logspace(
            jnp.log10(alpha_range[0]), jnp.log10(alpha_range[1]), n_alphas
        )

        gcv_scores = []
        train_errors = []
        effective_dof = []

        for alpha in alphas:
            alpha_val = float(alpha)

            # GCV score
            gcv = self._compute_gcv(alpha_val, s_sq, U, y, n_samples)
            gcv_scores.append(gcv)

            # Effective degrees of freedom
            df = float(jnp.sum(s_sq / (s_sq + alpha_val)))
            effective_dof.append(df)

            # Training error for this alpha
            # Compute weights: w = V @ (S/(S² + α)) @ (U.T @ y)
            Uty = U.T @ y
            shrinkage = S / (s_sq + alpha_val)
            w_alpha = Vt.T @ (shrinkage.reshape(-1, 1) * Uty)
            y_pred = Phi @ w_alpha
            train_err = float(jnp.mean((y - y_pred) ** 2))
            train_errors.append(train_err)

        # Find optimal alpha
        best_idx = jnp.argmin(jnp.array(gcv_scores))
        alpha_opt = float(alphas[best_idx])

        # Fit final model with optimal alpha
        self.alpha = alpha_opt
        if method == "standard":
            self._fit_standard(Phi, y)
        elif method == "bayesian":
            self._fit_bayesian(Phi, y)

        self.is_fitted = True
        self.method = method
        self.input_dim = input_dim
        self.alpha_opt = alpha_opt
        self.gcv_score = gcv_scores[best_idx]

        # Return full path information
        path_info = {
            "alphas": np.array(alphas),
            "gcv_scores": np.array(gcv_scores),
            "train_errors": np.array(train_errors),
            "effective_dof": np.array(effective_dof),
            "alpha_opt": alpha_opt,
            "gcv_opt": gcv_scores[best_idx],
            "dof_opt": effective_dof[best_idx],
        }

        return path_info

    def plot_gcv_path(self, path_info: dict, save_path: str = None):
        """
        Plot GCV regularization path.
        """
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Plot 1: GCV score vs alpha
        ax = axes[0, 0]
        ax.semilogx(
            path_info["alphas"], path_info["gcv_scores"], "b-", linewidth=2
        )
        ax.axvline(
            path_info["alpha_opt"],
            color="r",
            linestyle="--",
            label=f'Optimal α = {path_info["alpha_opt"]:.2e}',
        )
        ax.set_xlabel("Regularization α")
        ax.set_ylabel("GCV Score")
        ax.set_title("GCV Score vs Regularization")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 2: Training error vs alpha
        ax = axes[0, 1]
        ax.loglog(
            path_info["alphas"], path_info["train_errors"], "g-", linewidth=2
        )
        ax.axvline(path_info["alpha_opt"], color="r", linestyle="--")
        ax.set_xlabel("Regularization α")
        ax.set_ylabel("Training MSE")
        ax.set_title("Training Error vs Regularization")
        ax.grid(True, alpha=0.3)

        # Plot 3: Effective DOF vs alpha
        ax = axes[1, 0]
        ax.semilogx(
            path_info["alphas"], path_info["effective_dof"], "m-", linewidth=2
        )
        ax.axvline(path_info["alpha_opt"], color="r", linestyle="--")
        ax.axhline(
            path_info["dof_opt"],
            color="r",
            linestyle=":",
            label=f'DOF at optimum = {path_info["dof_opt"]:.1f}',
        )
        ax.set_xlabel("Regularization α")
        ax.set_ylabel("Effective Degrees of Freedom")
        ax.set_title("Model Complexity vs Regularization")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 4: GCV vs DOF
        ax = axes[1, 1]
        ax.plot(
            path_info["effective_dof"],
            path_info["gcv_scores"],
            "k-",
            linewidth=2,
        )
        ax.axvline(path_info["dof_opt"], color="r", linestyle="--")
        ax.set_xlabel("Effective Degrees of Freedom")
        ax.set_ylabel("GCV Score")
        ax.set_title("GCV vs Model Complexity")
        ax.grid(True, alpha=0.3)

        plt.suptitle(
            "GCV Regularization Path Analysis", fontsize=14, fontweight="bold"
        )
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")

        plt.show()
