# Authors: T. Moudiki
#
# License: BSD 3 Clause Clear

import numpy as np
import sklearn.metrics as skm2
from .glm import GLM
from ..utils import matrixops as mo
from sklearn.base import RegressorMixin
from ..optimizers import Optimizer


class GLMRegressor(GLM, RegressorMixin):
    """Generalized 'linear' models using quasi-randomized networks (regression)

    Attributes:

        n_hidden_features: int
            number of nodes in the hidden layer

        lambda1: float
            regularization parameter for GLM coefficients on original features

        alpha1: float
            controls compromize between l1 and l2 norm of GLM coefficients on original features

        lambda2: float
            regularization parameter for GLM coefficients on nonlinear features

        alpha2: float
            controls compromize between l1 and l2 norm of GLM coefficients on nonlinear features

        family: str
            "gaussian", "laplace" or "poisson" (for now)

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

        direct_link: boolean
            indicates if the original predictors are included (True) in model's
            fitting or not (False)

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

        optimizer: object
            optimizer, from class nnetsauce.utils.Optimizer

        seed: int
            reproducibility seed for nodes_sim=='uniform'

    Attributes:

        beta_: vector
            regression coefficients

    Examples:

    See [https://github.com/Techtonique/nnetsauce/blob/master/examples/glm_regression.py](https://github.com/Techtonique/nnetsauce/blob/master/examples/glm_regression.py)

    """

    # construct the object -----

    def __init__(
        self,
        n_hidden_features=5,
        lambda1=0.01,
        alpha1=0.5,
        lambda2=0.01,
        alpha2=0.5,
        family="gaussian",
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
        optimizer=Optimizer(),
        seed=123,
    ):
        super().__init__(
            n_hidden_features=n_hidden_features,
            lambda1=lambda1,
            alpha1=alpha1,
            lambda2=lambda2,
            alpha2=alpha2,
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
            optimizer=optimizer,
            seed=seed,
        )

        self.family = family

    def gaussian_loss(self, y, row_index, XB):
        return 0.5 * np.mean(np.square(y[row_index] - XB))

    def laplace_loss(self, y, row_index, XB):
        return 0.5 * np.mean(np.abs(y[row_index] - XB))

    def poisson_loss(self, y, row_index, XB):
        return -np.mean(y[row_index] * XB - np.exp(XB))

    def loss_func(
        self,
        beta,
        group_index,
        X,
        y,
        row_index=None,
        type_loss="gaussian",
        **kwargs
    ):
        res = {
            "gaussian": self.gaussian_loss,
            "laplace": self.laplace_loss,
            "poisson": self.poisson_loss,
        }

        if row_index is None:
            row_index = range(len(y))
            XB = self.compute_XB(X, beta=beta)

            return res[type_loss](y, row_index, XB) + self.compute_penalty(
                group_index=group_index, beta=beta
            )

        XB = self.compute_XB(X, beta=beta, row_index=row_index)

        return res[type_loss](y, row_index, XB) + self.compute_penalty(
            group_index=group_index, beta=beta
        )

    def fit(self, X, y, **kwargs):
        """Fit GLM model to training data (X, y).

        Args:

            X: {array-like}, shape = [n_samples, n_features]
                Training vectors, where n_samples is the number
                of samples and n_features is the number of features.

            y: array-like, shape = [n_samples]
                Target values.

            **kwargs: additional parameters to be passed to
                    self.cook_training_set or self.obj.fit

        Returns:

            self: object

        """

        self.beta_ = None

        self.n_iter = 0

        n, self.group_index = X.shape

        centered_y, scaled_Z = self.cook_training_set(y=y, X=X, **kwargs)

        n_Z = scaled_Z.shape[0]

        # initialization
        beta_ = np.linalg.lstsq(scaled_Z, centered_y, rcond=None)[0]

        # optimization
        # fit(self, loss_func, response, x0, **kwargs):
        # loss_func(self, beta, group_index, X, y,
        #          row_index=None, type_loss="gaussian",
        #          **kwargs)
        self.optimizer.fit(
            self.loss_func,
            response=centered_y,
            x0=beta_,
            group_index=self.group_index,
            X=scaled_Z,
            y=centered_y,
            type_loss=self.family,
            **kwargs
        )

        self.beta_ = self.optimizer.results[0]

        return self

    def predict(self, X, **kwargs):
        """Predict test data X.

        Args:

            X: {array-like}, shape = [n_samples, n_features]
                Training vectors, where n_samples is the number
                of samples and n_features is the number of features.

            **kwargs: additional parameters to be passed to
                    self.cook_test_set

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
                + np.dot(self.cook_test_set(new_X, **kwargs), self.beta_)
            )[0]

        return self.y_mean_ + np.dot(
            self.cook_test_set(X, **kwargs), self.beta_
        )
