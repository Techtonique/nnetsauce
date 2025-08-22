# Authors: Thierry Moudiki
#
# License: BSD 3

import numpy as np
from .base import Base
import sklearn.metrics as skm
from ..utils import matrixops as mo
from ..utils import lmfuncs as lmf
from ..utils import misc as mx
from sklearn.base import RegressorMixin


class BaseRegressor(Base, RegressorMixin):
    """Random Vector Functional Link Network regression without shrinkage

    Parameters:

        n_hidden_features: int
            number of nodes in the hidden layer

        activation_name: str
            activation function: 'relu', 'tanh', 'sigmoid', 'prelu' or 'elu'

        a: float
            hyperparameter for 'prelu' or 'elu' activation function

        nodes_sim: str
            type of simulation for hidden layer nodes: 'sobol', 'hammersley', 'halton',
            'uniform'

        bias: boolean
            indicates if the hidden layer contains a bias term (True) or
            not (False)

        dropout: float
            regularization parameter; (random) percentage of nodes dropped out
            of the training

        direct_link: boolean
            indicates if the original features are included (True) in model's
            fitting or not (False)

        n_clusters: int
            number of clusters for type_clust='kmeans' or type_clust='gmm'
            clustering (could be 0: no clustering)

        cluster_encode: bool
            defines how the variable containing clusters is treated (default is one-hot);
            if `False`, then labels are used, without one-hot encoding

        type_clust: str
            type of clustering method: currently k-means ('kmeans') or Gaussian
            Mixture Model ('gmm')

        type_scaling: a tuple of 3 strings
            scaling methods for inputs, hidden layer, and clustering respectively
            (and when relevant).
            Currently available: standardization ('std') or MinMax scaling ('minmax')

        col_sample: float
            percentage of features randomly chosen for training

        row_sample: float
            percentage of rows chosen for training, by stratified bootstrapping

        seed: int
            reproducibility seed for nodes_sim=='uniform', clustering and dropout

        backend: str
            "cpu" or "gpu" or "tpu"

    Attributes:

        beta_: vector
            regression coefficients

        GCV_: float
            Generalized Cross-Validation error

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

    def fit(self, X, y, **kwargs):
        """Fit BaseRegressor to training data (X, y)

        Parameters:

            X: {array-like}, shape = [n_samples, n_features]
                Training vectors, where n_samples is the number
                of samples and n_features is the number of features

            y: array-like, shape = [n_samples]
                Target values

            **kwargs: additional parameters to be passed to self.cook_training_set

        Returns:

            self: object
        """

        centered_y, scaled_Z = self.cook_training_set(y=y, X=X, **kwargs)

        fit_obj = lmf.beta_Sigma_hat(
            X=scaled_Z, y=centered_y, backend=self.backend
        )

        self.beta_ = fit_obj["beta_hat"]

        self.GCV_ = fit_obj["GCV"]

        return self

    def predict(self, X, **kwargs):
        """Predict test data X.

        Parameters:

            X: {array-like}, shape = [n_samples, n_features]
                Training vectors, where n_samples is the number
                of samples and n_features is the number of features

            **kwargs: additional parameters to be passed to self.cook_test_set

        Returns:

            model predictions: {array-like}
        """

        if len(X.shape) == 1:
            n_features = X.shape[0]
            new_X = mo.rbind(
                X.reshape(1, n_features),
                np.ones(n_features).reshape(1, n_features),
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
