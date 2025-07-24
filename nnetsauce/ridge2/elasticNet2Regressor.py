import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax import jit, grad, value_and_grad, lax
from sklearn.base import RegressorMixin
from functools import partial
from typing import Optional, Tuple, Union, Dict, Any
from .ridge2 import Ridge2


class ElasticNet2Regressor(Ridge2, RegressorMixin):
    """Enhanced Elastic Net with dual regularization paths - Fixed JAX implementation."""

    def __init__(
        self,
        n_hidden_features: int = 5,
        activation_name: str = "relu",
        a: float = 0.01,
        nodes_sim: str = "sobol",
        bias: bool = True,
        dropout: float = 0,
        n_clusters: int = 2,
        cluster_encode: bool = True,
        type_clust: str = "kmeans",
        type_scaling: Tuple[str, str, str] = ("std", "std", "std"),
        lambda1: float = 0.1,
        lambda2: float = 0.1,
        l1_ratio1: float = 0.5,
        l1_ratio2: float = 0.5,
        max_iter: int = 1000,
        tol: float = 1e-4,
        solver: str = "cd",
        learning_rate: float = 0.01,
        type_loss: str = "mse",
        quantile: float = 0.5,
        patience: int = 10,
        verbose: bool = False,
        seed: int = 123,
    ):
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
        )
        
        self.l1_ratio1 = l1_ratio1
        self.l1_ratio2 = l1_ratio2
        self.max_iter = max_iter
        self.tol = tol
        self.solver = solver
        self.learning_rate = learning_rate
        self.type_loss = type_loss
        self.quantile = quantile
        self.patience = patience
        self.verbose = verbose
        self.type_fit = "regression"
        
        self._validate_parameters()
        self.key = jax.random.PRNGKey(seed)
        self._init_jax_functions()

    def _validate_parameters(self) -> None:
        """Validate all input parameters."""
        if self.type_loss not in ["mse", "quantile"]:
            raise ValueError("type_loss must be 'mse' or 'quantile'")
        if self.type_loss == "quantile" and not (0 < self.quantile < 1):
            raise ValueError("quantile must be between 0 and 1")
        if self.solver not in ["cd"] and not hasattr(optax, self.solver):
            raise ValueError(f"Unknown solver: {self.solver}")
        if any(ratio < 0 or ratio > 1 for ratio in [self.l1_ratio1, self.l1_ratio2]):
            raise ValueError("l1_ratios must be between 0 and 1")

    def _init_jax_functions(self) -> None:
        """Initialize and JIT compile all key functions."""
        # Use static_argnums for functions that take n_direct
        self._jax_objective = jit(self._objective, static_argnums=(3,))
        self._jax_grad = jit(grad(self._objective), static_argnums=(3,))
        self._jax_value_and_grad = jit(value_and_grad(self._objective), static_argnums=(3,))
        self._jax_coordinate_descent_step = jit(self._coordinate_descent_step, static_argnums=(3,))
        self._jax_predict = jit(self._predict)

    def _penalty(self, beta: jnp.ndarray, n_direct: int) -> jnp.ndarray:
        """Compute elastic net penalty with static shapes."""
        # Split the array without dynamic slicing
        beta_direct = beta[:n_direct]
        beta_hidden = beta[n_direct:]
        
        l1_1 = self.lambda1 * self.l1_ratio1 * jnp.sum(jnp.abs(beta_direct))
        l2_1 = 0.5 * self.lambda1 * (1 - self.l1_ratio1) * jnp.sum(beta_direct**2)
        l1_2 = self.lambda2 * self.l1_ratio2 * jnp.sum(jnp.abs(beta_hidden))
        l2_2 = 0.5 * self.lambda2 * (1 - self.l1_ratio2) * jnp.sum(beta_hidden**2)
        
        return l1_1 + l2_1 + l1_2 + l2_2

    def _objective(self, beta: jnp.ndarray, X: jnp.ndarray, 
                 y: jnp.ndarray, n_direct: int) -> jnp.ndarray:
        """Compute objective function value."""
        residuals = y - jnp.dot(X, beta)
        
        if self.type_loss == "mse":
            loss = 0.5 * jnp.mean(residuals**2)
        else:  # quantile loss
            loss = jnp.mean(jnp.maximum(self.quantile * residuals, 
                                      (self.quantile - 1) * residuals))
        
        return loss + self._penalty(beta, n_direct)

    def _soft_threshold(self, x: jnp.ndarray, threshold: float) -> jnp.ndarray:
        """Soft thresholding operator for coordinate descent."""
        return jnp.sign(x) * jnp.maximum(jnp.abs(x) - threshold, 0)

    def _coordinate_descent_step(self, beta: jnp.ndarray, X: jnp.ndarray, 
                               y: jnp.ndarray, n_direct: int, j: int) -> jnp.ndarray:
        """Single coordinate descent step for feature j."""
        X_j = X[:, j]
        r = y - jnp.dot(X, beta) + X_j * beta[j]
        XtX_jj = jnp.dot(X_j, X_j)
        update = jnp.dot(X_j, r) / (XtX_jj + 1e-10)
        
        lambda_ = jnp.where(j < n_direct, self.lambda1, self.lambda2)
        l1_ratio = jnp.where(j < n_direct, self.l1_ratio1, self.l1_ratio2)
        
        beta_j = self._soft_threshold(update, lambda_ * l1_ratio)
        return beta.at[j].set(beta_j / (1 + lambda_ * (1 - l1_ratio)))

    def _coordinate_descent(self, X: jnp.ndarray, y: jnp.ndarray, 
                           n_direct: int) -> jnp.ndarray:
        """Coordinate descent optimization."""
        beta = jnp.zeros(X.shape[1])
        best_beta = beta
        best_loss = float('inf')
        patience_count = 0
        
        for i in range(self.max_iter):
            beta_old = beta.copy()
            
            # Update each coordinate sequentially
            for j in range(X.shape[1]):
                beta = self._jax_coordinate_descent_step(beta, X, y, n_direct, j)
            
            # Check convergence
            current_loss = self._jax_objective(beta, X, y, n_direct)
            
            if current_loss < best_loss - self.tol:
                best_beta = beta
                best_loss = current_loss
                patience_count = 0
            else:
                patience_count += 1
                
            if self.verbose and i % 10 == 0:
                print(f"Iter {i}: Loss {current_loss:.4f}")
                
            if patience_count >= self.patience:
                if self.verbose:
                    print(f"Early stopping at iter {i}")
                break
                
            # Additional convergence check
            if jnp.max(jnp.abs(beta - beta_old)) < self.tol:
                if self.verbose:
                    print(f"Converged at iter {i}")
                break
                
        return best_beta

    def _get_optimizer(self) -> optax.GradientTransformation:
        """Get optax optimizer based on solver name."""
        try:
            optimizer_fn = getattr(optax, self.solver)
            return optimizer_fn(learning_rate=self.learning_rate)
        except AttributeError:
            raise ValueError(f"Unknown optimizer: {self.solver}")

    def _optax_optimize(self, X: jnp.ndarray, y: jnp.ndarray, 
                       n_direct: int) -> jnp.ndarray:
        """Optimization using optax optimizers."""
        optimizer = self._get_optimizer()
        beta = jnp.zeros(X.shape[1])
        opt_state = optimizer.init(beta)
        
        @jit
        def update_step(beta, opt_state):
            loss, grads = self._jax_value_and_grad(beta, X, y, n_direct)
            updates, new_opt_state = optimizer.update(grads, opt_state, beta)
            new_beta = optax.apply_updates(beta, updates)
            return new_beta, new_opt_state, loss
        
        best_beta = beta
        best_loss = float('inf')
        patience_count = 0
        
        for i in range(self.max_iter):
            beta, opt_state, current_loss = update_step(beta, opt_state)
            
            if current_loss < best_loss - self.tol:
                best_beta = beta
                best_loss = current_loss
                patience_count = 0
            else:
                patience_count += 1
                
            if self.verbose and i % 10 == 0:
                print(f"Iter {i}: Loss {current_loss:.4f}")
                
            if patience_count >= self.patience:
                if self.verbose:
                    print(f"Early stopping at iter {i}")
                break
                
        return best_beta

    def fit(self, X: Union[jnp.ndarray, np.ndarray], 
            y: Union[jnp.ndarray, np.ndarray], **kwargs) -> "ElasticNet2Regressor":
        """Fit model with selected optimization method."""
        X = jnp.asarray(X)
        y = jnp.asarray(y)
        
        centered_y, scaled_Z = self.cook_training_set(y=y, X=X, **kwargs)
        centered_y = jnp.asarray(centered_y)
        scaled_Z = jnp.asarray(scaled_Z)
        
        n_X, p_X = X.shape
        n_Z, p_Z = scaled_Z.shape

        n_direct = (p_X + (self.n_clusters if self.cluster_encode else 1) 
                   if self.n_clusters > 0 else p_X)

        X_ = scaled_Z[:, :n_direct]
        Phi_X_ = scaled_Z[:, n_direct:p_Z]
        all_features = jnp.hstack([X_, Phi_X_])
        
        if self.solver == "cd":
            self.beta_ = self._coordinate_descent(all_features, centered_y, n_direct)
        else:
            self.beta_ = self._optax_optimize(all_features, centered_y, n_direct)
        
        self.y_mean_ = jnp.mean(y)
        return self

    def _predict(self, X: jnp.ndarray) -> jnp.ndarray:
        """JAX implementation of prediction."""
        return self.y_mean_ + jnp.dot(X, self.beta_)

    def predict(self, X: Union[jnp.ndarray, np.ndarray], 
               **kwargs) -> jnp.ndarray:
        """Predict using fitted model."""
        X = jnp.asarray(X)
        
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
            
        test_features = jnp.asarray(self.cook_test_set(X, **kwargs))
        return self._jax_predict(test_features)

    def score(self, X: Union[jnp.ndarray, np.ndarray], 
              y: Union[jnp.ndarray, np.ndarray], **kwargs) -> float:
        """Compute R² score."""
        y_pred = self.predict(X, **kwargs)
        y = jnp.asarray(y)
        
        ss_res = jnp.sum((y - y_pred) ** 2)
        ss_tot = jnp.sum((y - jnp.mean(y)) ** 2)
        return float(1 - (ss_res / (ss_tot + 1e-10)))

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Get parameters for this estimator."""
        params = super().get_params(deep=deep)
        params.update({
            'l1_ratio1': self.l1_ratio1,
            'l1_ratio2': self.l1_ratio2,
            'max_iter': self.max_iter,
            'tol': self.tol,
            'solver': self.solver,
            'learning_rate': self.learning_rate,
            'type_loss': self.type_loss,
            'quantile': self.quantile,
            'patience': self.patience,
            'verbose': self.verbose,
        })
        return params

    def set_params(self, **params) -> "ElasticNet2Regressor":
        """Set parameters for this estimator."""
        super().set_params(**params)
        self._validate_parameters()
        self._init_jax_functions()
        return self