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
    from jax import jit, grad
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False


class ElasticNet2Regressor(Ridge2, RegressorMixin):
    """Enhanced Elastic Net with dual regularization paths, JAX support, and coordinate descent.
    
    Features:
    - Separate L1/L2 ratios for direct (lambda1/l1_ratio1) and hidden (lambda2/l1_ratio2) paths
    - JAX acceleration for GPU/TPU when backend != 'cpu'
    - Choice of optimization methods (L-BFGS-B or coordinate descent)

    Parameters:
        n_hidden_features: int
            Number of nodes in the hidden layer
        activation_name: str
            Activation function: 'relu', 'tanh', 'sigmoid', 'prelu' or 'elu'
        a: float
            Hyperparameter for 'prelu' or 'elu' activation
        nodes_sim: str
            Node simulation type: 'sobol', 'hammersley', 'halton', 'uniform'
        bias: bool
            Whether to include bias term in hidden layer
        dropout: float
            Dropout rate (regularization)
        n_clusters: int
            Number of clusters (0 for no clustering)
        cluster_encode: bool
            Whether to one-hot encode clusters
        type_clust: str
            Clustering method: 'kmeans' or 'gmm'
        type_scaling: tuple
            Scaling methods for (inputs, hidden layer, clusters)
        lambda1: float
            Regularization strength for direct connections
        lambda2: float
            Regularization strength for hidden layer
        l1_ratio1: float
            L1 ratio (0-1) for direct connections
        l1_ratio2: float
            L1 ratio (0-1) for hidden layer
        max_iter: int
            Maximum optimization iterations
        tol: float
            Optimization tolerance
        solver: str
            Optimization method: 'lbfgs' or 'cd' (coordinate descent)
        seed: int
            Random seed
        backend: str
            'cpu', 'gpu', or 'tpu'
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
        l1_ratio1=0.5,
        l1_ratio2=0.5,
        max_iter=1000,
        tol=1e-4,
        solver="lbfgs",
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
            n_clusters=n_clusters,
            cluster_encode=cluster_encode,
            type_clust=type_clust,
            type_scaling=type_scaling,
            lambda1=lambda1,
            lambda2=lambda2,
            seed=seed,
            backend=backend,
        )
        
        self.l1_ratio1 = l1_ratio1
        self.l1_ratio2 = l1_ratio2
        self.max_iter = max_iter
        self.tol = tol
        self.solver = solver
        self.type_fit = "regression"
        
        # Initialize JAX-related attributes
        self._jax_initialized = False
        self._init_jax_functions()

    def _init_jax_functions(self):
        """Initialize JAX functions if backend is not CPU and JAX is available"""
        if self.backend != "cpu" and JAX_AVAILABLE and not self._jax_initialized:
            # JIT compile key functions
            self._jax_elastic_net_penalty = jit(self._jax_penalty)
            self._jax_objective = jit(self._jax_obj)
            self._jax_grad = jit(grad(self._jax_obj))
            self._jax_initialized = True

    def _jax_penalty(self, beta, n_direct):
        """JAX version of elastic net penalty"""
        beta_direct = beta[:n_direct]
        beta_hidden = beta[n_direct:]
        
        l1_1 = self.lambda1 * self.l1_ratio1 * jnp.sum(jnp.abs(beta_direct))
        l2_1 = 0.5 * self.lambda1 * (1-self.l1_ratio1) * jnp.sum(beta_direct**2)
        l1_2 = self.lambda2 * self.l1_ratio2 * jnp.sum(jnp.abs(beta_hidden))
        l2_2 = 0.5 * self.lambda2 * (1-self.l1_ratio2) * jnp.sum(beta_hidden**2)
        
        return l1_1 + l2_1 + l1_2 + l2_2

    def _jax_obj(self, beta, X, y, n_direct):
        """JAX version of objective function"""
        residuals = y - jnp.dot(X, beta)
        mse = jnp.mean(residuals**2)
        penalty = self._jax_penalty(beta, n_direct)
        return 0.5 * mse + penalty

    def _numpy_penalty(self, beta, n_direct):
        """NumPy version of elastic net penalty"""
        beta_direct = beta[:n_direct]
        beta_hidden = beta[n_direct:]
        
        l1_1 = self.lambda1 * self.l1_ratio1 * np.sum(np.abs(beta_direct))
        l2_1 = 0.5 * self.lambda1 * (1-self.l1_ratio1) * np.sum(beta_direct**2)
        l1_2 = self.lambda2 * self.l1_ratio2 * np.sum(np.abs(beta_hidden))
        l2_2 = 0.5 * self.lambda2 * (1-self.l1_ratio2) * np.sum(beta_hidden**2)
        
        return l1_1 + l2_1 + l1_2 + l2_2

    def _numpy_obj(self, beta, X, y, n_direct):
        """NumPy version of objective function"""
        residuals = y - np.dot(X, beta)
        mse = np.mean(residuals**2)
        penalty = self._numpy_penalty(beta, n_direct)
        return 0.5 * mse + penalty

    def _soft_threshold(self, x, threshold):
        """Soft thresholding operator for coordinate descent"""
        return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)

    def _coordinate_descent(self, X, y, n_direct):
        """Coordinate descent optimization"""
        n_samples, n_features = X.shape
        beta = np.zeros(n_features)
        XtX = X.T @ X
        Xty = X.T @ y
        diag_XtX = np.diag(XtX)
        
        for _ in range(self.max_iter):
            beta_old = beta.copy()
            
            for j in range(n_features):
                # Compute partial residual
                X_j = X[:, j]
                r = y - X @ beta + X_j * beta[j]
                
                # Compute unregularized update
                update = X_j @ r / (diag_XtX[j] + 1e-10)
                
                # Apply appropriate regularization
                if j < n_direct:  # Direct connection
                    lambda_ = self.lambda1
                    l1_ratio = self.l1_ratio1
                else:  # Hidden layer connection
                    lambda_ = self.lambda2
                    l1_ratio = self.l1_ratio2
                
                # Apply soft thresholding for L1 and shrinkage for L2
                beta[j] = self._soft_threshold(update, lambda_ * l1_ratio)
                beta[j] /= (1 + lambda_ * (1 - l1_ratio))
            
            # Check convergence
            if np.max(np.abs(beta - beta_old)) < self.tol:
                break
                
        return beta

    def fit(self, X, y, **kwargs):
        """Fit model with selected optimization method"""
        centered_y, scaled_Z = self.cook_training_set(y=y, X=X, **kwargs)
        n_X, p_X = X.shape
        n_Z, p_Z = scaled_Z.shape

        if self.n_clusters > 0:
            n_direct = p_X + (self.n_clusters if self.cluster_encode else 1)
        else:
            n_direct = p_X

        X_ = scaled_Z[:, 0:n_direct]
        Phi_X_ = scaled_Z[:, n_direct:p_Z]
        all_features = np.hstack([X_, Phi_X_])
        
        # Convert to JAX arrays if using GPU/TPU
        if self.backend != "cpu" and JAX_AVAILABLE:
            all_features = jnp.array(all_features)
            centered_y = jnp.array(centered_y)
            beta_init = jnp.zeros(all_features.shape[1])
            
            if self.solver == "lbfgs":
                res = minimize(
                    fun=self._jax_obj,
                    x0=beta_init,
                    args=(all_features, centered_y, n_direct),
                    method='L-BFGS-B',
                    jac=self._jax_grad,
                    options={'maxiter': self.max_iter, 'gtol': self.tol}
                )
                self.beta_ = np.array(res.x)
            else:
                # Fall back to NumPy for coordinate descent
                self.beta_ = self._coordinate_descent(
                    np.array(all_features), 
                    np.array(centered_y), 
                    n_direct
                )
        else:
            # NumPy backend
            beta_init = np.zeros(all_features.shape[1])
            
            if self.solver == "cd":
                self.beta_ = self._coordinate_descent(
                    all_features, 
                    centered_y, 
                    n_direct
                )
            else:
                res = minimize(
                    fun=self._numpy_obj,
                    x0=beta_init,
                    args=(all_features, centered_y, n_direct),
                    method='L-BFGS-B',
                    options={'maxiter': self.max_iter, 'gtol': self.tol}
                )
                self.beta_ = res.x
        
        self.y_mean_ = np.mean(y)
        return self

    def predict(self, X, **kwargs):
        """Predict using fitted model"""
        if len(X.shape) == 1:
            n_features = X.shape[0]
            new_X = mo.rbind(
                x=X.reshape(1, n_features),
                y=np.ones(n_features).reshape(1, n_features),
                backend=self.backend,
            )
            return (
                self.y_mean_
                + mo.safe_sparse_dot(
                    a=self.cook_test_set(new_X, **kwargs),
                    b=self.beta_,
                    backend=self.backend,
                )
            )[0]

        return self.y_mean_ + mo.safe_sparse_dot(
            a=self.cook_test_set(X, **kwargs),
            b=self.beta_,
            backend=self.backend,
        )
