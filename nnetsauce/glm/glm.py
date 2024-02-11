"""Generalized 'linear' models for quasi-randomized networks (nonlinear models!)"""

# Authors: Thierry Moudiki
#
# License: BSD 3 Clear


import numpy as np
from ..base import Base
from ..optimizers import Optimizer


class GLM(Base):
    """Generalized 'linear' models for quasi-randomized networks (nonlinear models!)

    Parameters:

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
            reproducibility seed for nodes_sim=='uniform' and clustering

    Attributes:

        beta_: vector
            regression coefficients
    """

    # construct the object -----

    def __init__(
        self,
        n_hidden_features=5,
        lambda1=0.01,
        alpha1=0.5,
        lambda2=0.01,
        alpha2=0.5,
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
        )

        self.lambda1 = lambda1
        self.alpha1 = alpha1
        self.lambda2 = lambda2
        self.alpha2 = alpha2
        self.optimizer = optimizer
        self.beta_ = None

    def compute_XB(self, X, beta=None, row_index=None):
        if beta is not None:
            if row_index is None:
                return np.dot(X, beta)

            return np.dot(X[row_index, :], beta)

        # self.beta_ is None in this case
        if row_index is None:
            return np.dot(X, self.beta_)

        return np.dot(X[row_index, :], self.beta_)

    def compute_XB2(self, X, beta=None, row_index=None):
        def f00(X):
            return np.dot(X, self.beta_)

        def f01(X):
            return np.dot(X[row_index, :], self.beta_)

        def f11(X):
            return np.dot(X[row_index, :], beta)

        def f10(X):
            return np.dot(X, beta)

        h_result = {"00": f00, "01": f01, "11": f11, "10": f10}

        result_code = str(0 if beta is None else 1)
        result_code += str(0 if row_index is None else 1)

        return h_result[result_code](X)

    def penalty(self, beta1, beta2, lambda1, lambda2, alpha1, alpha2):
        res = lambda1 * (
            0.5 * (1 - alpha1) * np.sum(np.square(beta1))
            + alpha1 * np.sum(np.abs(beta1))
        )
        res += lambda2 * (
            0.5 * (1 - alpha2) * np.sum(np.square(beta2))
            + alpha2 * np.sum(np.abs(beta2))
        )

        return res

    def compute_penalty(self, group_index, beta):
        return self.penalty(
            beta1=beta[0:group_index],
            beta2=beta[group_index: len(beta)],
            lambda1=self.lambda1,
            lambda2=self.lambda2,
            alpha1=self.alpha1,
            alpha2=self.alpha2,
        )
