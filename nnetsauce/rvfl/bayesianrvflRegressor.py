# Authors: Thierry Moudiki
#
# License: BSD 3

import numpy as np
import sklearn.metrics as skm
from ..base import Base
from ..utils import misc as mx
from ..utils import matrixops as mo
from ..utils import lmfuncs as lmf
from sklearn.base import RegressorMixin


class BayesianRVFLRegressor(Base, RegressorMixin):
    """Bayesian Random Vector Functional Link Network regression with one prior

    Parameters:

        n_hidden_features: int
            number of nodes in the hidden layer

        activation_name: str
            activation function: 'relu', 'tanh', 'sigmoid', 'prelu' or 'elu'

        a: float
            hyperparameter for 'prelu' or 'elu' activation function

        nodes_sim: str
            type of simulation for the nodes: 'sobol', 'hammersley', 'halton', 'uniform'

        bias: boolean
            indicates if the hidden layer contains a bias term (True) or not (False)

        dropout: float
            regularization parameter; (random) percentage of nodes dropped out
            of the training

        direct_link: boolean
            indicates if the original features are included (True) in model''s fitting or not (False)

        n_clusters: int
            number of clusters for 'kmeans' or 'gmm' clustering (could be 0: no clustering)

        cluster_encode: bool
            defines how the variable containing clusters is treated (default is one-hot)
            if `False`, then labels are used, without one-hot encoding

        type_clust: str
            type of clustering method: currently k-means ('kmeans') or Gaussian Mixture Model ('gmm')

        type_scaling: a tuple of 3 strings
            scaling methods for inputs, hidden layer, and clustering respectively
            (and when relevant).
            Currently available: standardization ('std') or MinMax scaling ('minmax')

        seed: int
            reproducibility seed for nodes_sim=='uniform'

        s: float
            std. dev. of regression parameters in Bayesian Ridge Regression

        sigma: float
            std. dev. of residuals in Bayesian Ridge Regression

        return_std: boolean
            if True, uncertainty around predictions is evaluated

        backend: str
            "cpu" or "gpu" or "tpu"

    Attributes:

        beta_: array-like
            regression''s coefficients

        Sigma_: array-like
            covariance of the distribution of fitted parameters

        GCV_: float
            Generalized cross-validation error

        y_mean_: float
            average response

    Examples:

    ```python
    TBD
    ```

    """

    # construct the object -----

    def __init__(
        self,
        n_hidden_features=5,
        activation_name="relu",
        a=0.01,
        nodes_sim="sobol",
        bias=True,
        dropout=0,
        direct_link=True,
        n_clusters=2,
        cluster_encode=True,
        type_clust="kmeans",
        type_scaling=("std", "std", "std"),
        seed=123,
        s=0.1,
        sigma=0.05,
        return_std=True,
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
            seed=seed,
            backend=backend,
        )
        self.s = s
        self.sigma = sigma
        self.beta_ = None
        self.Sigma_ = None
        self.GCV_ = None
        self.return_std = return_std

    def fit(self, X, y, **kwargs):
        """Fit BayesianRVFLRegressor to training data (X, y).

        Parameters:

            X: {array-like}, shape = [n_samples, n_features]
                Training vectors, where n_samples is the number
                of samples and n_features is the number of features.

            y: array-like, shape = [n_samples]
                Target values.

            **kwargs: additional parameters to be passed to
                    self.cook_training_set

        Returns:

            self: object

        """

        centered_y, scaled_Z = self.cook_training_set(y=y, X=X, **kwargs)

        fit_obj = lmf.beta_Sigma_hat_rvfl(
            X=scaled_Z,
            y=centered_y,
            s=self.s,
            sigma=self.sigma,
            fit_intercept=False,
            return_cov=self.return_std,
            backend=self.backend,
        )

        self.beta_ = fit_obj["beta_hat"]

        if self.return_std == True:
            self.Sigma_ = fit_obj["Sigma_hat"]

        self.GCV_ = fit_obj["GCV"]

        return self

    def predict(self, X, return_std=False, **kwargs):
        """Predict test data X.

        Parameters:

            X: {array-like}, shape = [n_samples, n_features]
                Training vectors, where n_samples is the number
                of samples and n_features is the number of features.

            return_std: {boolean}, standard dev. is returned or not

            **kwargs: additional parameters to be passed to
                    self.cook_test_set

        Returns:

            model predictions: {array-like}

        """

        if len(X.shape) == 1:  # one observation in the test set only
            n_features = X.shape[0]
            new_X = mo.rbind(
                x=X.reshape(1, n_features),
                y=np.ones(n_features).reshape(1, n_features),
                backend=self.backend,
            )

        self.return_std = return_std

        if self.return_std == False:
            if len(X.shape) == 1:
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

        else:  # confidence interval required for preds?
            if len(X.shape) == 1:
                Z = self.cook_test_set(new_X, **kwargs)

                pred_obj = lmf.beta_Sigma_hat_rvfl(
                    s=self.s,
                    sigma=self.sigma,
                    X_star=Z,
                    return_cov=True,
                    beta_hat_=self.beta_,
                    Sigma_hat_=self.Sigma_,
                    backend=self.backend,
                )

                return (
                    self.y_mean_ + pred_obj["preds"][0],
                    pred_obj["preds_std"][0],
                )

            Z = self.cook_test_set(X, **kwargs)

            pred_obj = lmf.beta_Sigma_hat_rvfl(
                s=self.s,
                sigma=self.sigma,
                X_star=Z,
                return_cov=True,
                beta_hat_=self.beta_,
                Sigma_hat_=self.Sigma_,
                backend=self.backend,
            )

            return (self.y_mean_ + pred_obj["preds"], pred_obj["preds_std"])
