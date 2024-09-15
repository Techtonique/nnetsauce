# Authors: Thierry Moudiki
#
# License: BSD 3 Clear

import numpy as np
import sklearn.metrics as skm2
from .glm import GLM
from ..utils import matrixops as mo
from ..utils import misc as mx
from sklearn.base import ClassifierMixin
from ..optimizers import Optimizer
from scipy.special import logsumexp, expit, erf


class GLMClassifier(GLM, ClassifierMixin):
    """Generalized 'linear' models using quasi-randomized networks (classification)

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
            optimizer, from class nnetsauce.Optimizer

        seed: int
            reproducibility seed for nodes_sim=='uniform'

    Attributes:

        beta_: vector
            regression coefficients

    Examples:

    See [https://github.com/Techtonique/nnetsauce/blob/master/examples/glm_classification.py](https://github.com/Techtonique/nnetsauce/blob/master/examples/glm_classification.py)

    """

    # construct the object -----

    def __init__(
        self,
        n_hidden_features=5,
        lambda1=0.01,
        alpha1=0.5,
        lambda2=0.01,
        alpha2=0.5,
        family="expit",
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

    def logit_loss(self, Y, row_index, XB):
        self.n_classes = Y.shape[1]  # len(np.unique(y))
        # Y = mo.one_hot_encode2(y, self.n_classes)
        # Y = self.optimizer.one_hot_encode(y, self.n_classes)

        # max_double = 709.0 # only if softmax
        # XB[XB > max_double] = max_double
        XB[XB > 709.0] = 709.0

        if row_index is None:
            return -np.mean(np.sum(Y * XB, axis=1) - logsumexp(XB))

        return -np.mean(np.sum(Y[row_index, :] * XB, axis=1) - logsumexp(XB))

    def expit_erf_loss(self, Y, row_index, XB):
        # self.n_classes = len(np.unique(y))
        # Y = mo.one_hot_encode2(y, self.n_classes)
        # Y = self.optimizer.one_hot_encode(y, self.n_classes)
        self.n_classes = Y.shape[1]

        if row_index is None:
            return -np.mean(np.sum(Y * XB, axis=1) - logsumexp(XB))

        return -np.mean(np.sum(Y[row_index, :] * XB, axis=1) - logsumexp(XB))

    def loss_func(
        self,
        beta,
        group_index,
        X,
        Y,
        y,
        row_index=None,
        type_loss="logit",
        **kwargs
    ):
        res = {
            "logit": self.logit_loss,
            "expit": self.expit_erf_loss,
            "erf": self.expit_erf_loss,
        }

        if row_index is None:
            row_index = range(len(y))
            XB = self.compute_XB(
                X,
                beta=np.reshape(beta, (X.shape[1], self.n_classes), order="F"),
            )

            return res[type_loss](Y, row_index, XB) + self.compute_penalty(
                group_index=group_index, beta=beta
            )

        XB = self.compute_XB(
            X,
            beta=np.reshape(beta, (X.shape[1], self.n_classes), order="F"),
            row_index=row_index,
        )

        return res[type_loss](Y, row_index, XB) + self.compute_penalty(
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

        assert mx.is_factor(
            y
        ), "y must contain only integers"  # change is_factor and subsampling everywhere

        self.classes_ = np.unique(y)  # for compatibility with sklearn
        self.n_classes_ = len(self.classes_)  # for compatibility with sklearn

        self.beta_ = None

        n, p = X.shape

        self.group_index = n * X.shape[1]

        self.n_classes = len(np.unique(y))

        output_y, scaled_Z = self.cook_training_set(y=y, X=X, **kwargs)

        # Y = mo.one_hot_encode2(output_y, self.n_classes)
        Y = self.optimizer.one_hot_encode(output_y, self.n_classes)

        # initialization
        beta_ = np.linalg.lstsq(scaled_Z, Y, rcond=None)[0]

        # optimization
        # fit(self, loss_func, response, x0, **kwargs):
        # loss_func(self, beta, group_index, X, y,
        #          row_index=None, type_loss="gaussian",
        #          **kwargs)
        self.optimizer.fit(
            self.loss_func,
            response=y,
            x0=beta_.flatten(order="F"),
            group_index=self.group_index,
            X=scaled_Z,
            Y=Y,
            y=y,
            type_loss=self.family,
        )

        self.beta_ = self.optimizer.results[0]
        self.classes_ = np.unique(y)

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

        return np.argmax(self.predict_proba(X, **kwargs), axis=1)

    def predict_proba(self, X, **kwargs):
        """Predict probabilities for test data X.

        Args:

            X: {array-like}, shape = [n_samples, n_features]
                Training vectors, where n_samples is the number
                of samples and n_features is the number of features.

            **kwargs: additional parameters to be passed to
                    self.cook_test_set

        Returns:

            probability estimates for test data: {array-like}

        """
        if len(X.shape) == 1:
            n_features = X.shape[0]
            new_X = mo.rbind(
                X.reshape(1, n_features),
                np.ones(n_features).reshape(1, n_features),
            )

            Z = self.cook_test_set(new_X, **kwargs)

        else:
            Z = self.cook_test_set(X, **kwargs)

        ZB = mo.safe_sparse_dot(
            Z,
            self.beta_.reshape(
                self.n_classes,
                X.shape[1] + self.n_hidden_features + self.n_clusters,
            ).T,
        )

        if self.family == "logit":
            exp_ZB = np.exp(ZB)

            return exp_ZB / exp_ZB.sum(axis=1)[:, None]

        if self.family == "expit":
            exp_ZB = expit(ZB)

            return exp_ZB / exp_ZB.sum(axis=1)[:, None]

        if self.family == "erf":
            exp_ZB = 0.5 * (1 + erf(ZB))

            return exp_ZB / exp_ZB.sum(axis=1)[:, None]
