# Authors: Thierry Moudiki
#
# License: BSD 3 Clear

import numpy as np
import platform
from scipy.optimize import minimize
import sklearn.metrics as skm2
from .ridge2 import Ridge2
from ..utils import matrixops as mo
from ..utils import misc as mx
from sklearn.base import ClassifierMixin
from scipy.special import logsumexp
from scipy.linalg import pinv

try:
    from jax.numpy.linalg import pinv as jpinv
except ImportError:
    pass


class Ridge2MultitaskClassifier(Ridge2, ClassifierMixin):
    """Multitask Ridge classification with 2 regularization parameters

    Parameters:

        n_hidden_features: int
            number of nodes in the hidden layer

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

        lambda1: float
            regularization parameter on direct link

        lambda2: float
            regularization parameter on hidden layer

        seed: int
            reproducibility seed for nodes_sim=='uniform'

        backend: str
            "cpu" or "gpu" or "tpu"

    Attributes:

        beta_: {array-like}
            regression coefficients

    Examples:

    See also [https://github.com/Techtonique/nnetsauce/blob/master/examples/ridgemtask_classification.py](https://github.com/Techtonique/nnetsauce/blob/master/examples/ridgemtask_classification.py)

    ```python
    import nnetsauce as ns
    import numpy as np
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    from sklearn import metrics
    from time import time

    breast_cancer = load_breast_cancer()
    Z = breast_cancer.data
    t = breast_cancer.target
    np.random.seed(123)
    X_train, X_test, y_train, y_test = train_test_split(Z, t, test_size=0.2)

    fit_obj = ns.Ridge2MultitaskClassifier(n_hidden_features=int(9.83730469e+01),
                                    dropout=4.31054687e-01,
                                    n_clusters=int(1.71484375e+00),
                                    lambda1=1.24023438e+01, lambda2=7.30263672e+03)

    start = time()
    fit_obj.fit(X_train, y_train)
    print(f"Elapsed {time() - start}")

    print(fit_obj.score(X_test, y_test))
    print(fit_obj.score(X_test, y_test, scoring="roc_auc"))

    start = time()
    preds = fit_obj.predict(X_test)
    print(f"Elapsed {time() - start}")
    print(metrics.classification_report(preds, y_test))
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
        n_clusters=2,
        cluster_encode=True,
        type_clust="kmeans",
        type_scaling=("std", "std", "std"),
        lambda1=0.1,
        lambda2=0.1,
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

        self.type_fit = "classification"

    def fit(self, X, y, **kwargs):
        """Fit Ridge model to training data (X, y).

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

        sys_platform = platform.system()

        assert mx.is_factor(y), "y must contain only integers"

        self.classes_ = np.unique(y)  # for compatibility with sklearn
        self.n_classes_ = len(self.classes_)  # for compatibility with sklearn

        output_y, scaled_Z = self.cook_training_set(y=y, X=X, **kwargs)

        n_X, p_X = X.shape
        n_Z, p_Z = scaled_Z.shape

        self.n_classes = len(np.unique(y))

        # multitask response
        Y = mo.one_hot_encode2(output_y, self.n_classes)

        if self.n_clusters > 0:
            if self.encode_clusters == True:
                n_features = p_X + self.n_clusters
            else:
                n_features = p_X + 1
        else:
            n_features = p_X

        X_ = scaled_Z[:, 0:n_features]
        Phi_X_ = scaled_Z[:, n_features:p_Z]

        B = mo.crossprod(x=X_, backend=self.backend) + self.lambda1 * np.diag(
            np.repeat(1, X_.shape[1])
        )
        C = mo.crossprod(x=Phi_X_, y=X_, backend=self.backend)
        D = mo.crossprod(
            x=Phi_X_, backend=self.backend
        ) + self.lambda2 * np.diag(np.repeat(1, Phi_X_.shape[1]))

        if sys_platform in ("Linux", "Darwin"):
            B_inv = pinv(B) if self.backend == "cpu" else jpinv(B)
        else:
            B_inv = pinv(B)

        W = mo.safe_sparse_dot(a=C, b=B_inv, backend=self.backend)
        S_mat = D - mo.tcrossprod(x=W, y=C, backend=self.backend)

        if sys_platform in ("Linux", "Darwin"):
            S_inv = pinv(S_mat) if self.backend == "cpu" else jpinv(S_mat)
        else:
            S_inv = pinv(S_mat)

        Y2 = mo.safe_sparse_dot(a=S_inv, b=W, backend=self.backend)
        inv = mo.rbind(
            mo.cbind(
                x=B_inv + mo.crossprod(x=W, y=Y2, backend=self.backend),
                y=-np.transpose(Y2),
                backend=self.backend,
            ),
            mo.cbind(x=-Y2, y=S_inv, backend=self.backend),
            backend=self.backend,
        )

        self.beta_ = mo.safe_sparse_dot(
            a=inv,
            b=mo.crossprod(x=scaled_Z, y=Y, backend=self.backend),
            backend=self.backend,
        )
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
                x=X.reshape(1, n_features),
                y=np.ones(n_features).reshape(1, n_features),
                backend=self.backend,
            )

            Z = self.cook_test_set(new_X, **kwargs)

        else:
            Z = self.cook_test_set(X, **kwargs)

        ZB = mo.safe_sparse_dot(a=Z, b=self.beta_, backend=self.backend)

        exp_ZB = np.exp(ZB)

        return exp_ZB / exp_ZB.sum(axis=1)[:, None]
