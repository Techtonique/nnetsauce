# Authors: Thierry Moudiki
#
# License: BSD 3

import numpy as np
import pickle
import sklearn.metrics as skm2
from ..utils import matrixops as mo
from ..utils import misc as mx
from ..base import Base
from scipy.special import expit
from sklearn.base import ClassifierMixin


class MultitaskClassifier(Base, ClassifierMixin):
    """Multitask Classification model based on regression models, with shared covariates

    Parameters:

        obj: object
            any object (must be a regression model) containing a method fit (obj.fit())
            and a method predict (obj.predict())

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

        col_sample: float
            percentage of covariates randomly chosen for training

        row_sample: float
            percentage of rows chosen for training, by stratified bootstrapping

        seed: int
            reproducibility seed for nodes_sim=='uniform'

        backend: str
            "cpu" or "gpu" or "tpu"

    Attributes:

        fit_objs_: dict
            objects adjusted to each individual time series

        n_classes_: int
            number of classes for the classifier

    Examples:

    See also [https://github.com/Techtonique/nnetsauce/blob/master/examples/mtask_classification.py](https://github.com/Techtonique/nnetsauce/blob/master/examples/mtask_classification.py)

    ```python
    import nnetsauce as ns
    import numpy as np
    from sklearn.datasets import load_breast_cancer
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split
    from sklearn import metrics
    from time import time

    breast_cancer = load_breast_cancer()
    Z = breast_cancer.data
    t = breast_cancer.target

    X_train, X_test, y_train, y_test = train_test_split(Z, t, test_size=0.2,
                                                        random_state=123+2*10)

    # Linear Regression is used
    regr = LinearRegression()
    fit_obj = ns.MultitaskClassifier(regr, n_hidden_features=5,
                                n_clusters=2, type_clust="gmm")

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
        obj,
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

        self.type_fit = "classification"
        self.obj = obj
        self.fit_objs_ = {}

    def fit(self, X, y, sample_weight=None, **kwargs):
        """Fit MultitaskClassifier to training data (X, y).

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

        assert mx.is_factor(y), "y must contain only integers"

        output_y, scaled_Z = self.cook_training_set(y=y, X=X, **kwargs)

        self.classes_ = np.unique(y)  # for compatibility with sklearn
        self.n_classes_ = len(self.classes_)  # for compatibility with sklearn

        # multitask response
        Y = mo.one_hot_encode2(output_y, self.n_classes_)

        # if sample_weight is None:
        for i in range(self.n_classes_):
            self.fit_objs_[i] = pickle.loads(
                pickle.dumps(self.obj.fit(scaled_Z, Y[:, i], **kwargs), -1)
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

        shape_X = X.shape

        probs = np.zeros((shape_X[0], self.n_classes_))

        if len(shape_X) == 1:
            n_features = shape_X[0]

            new_X = mo.rbind(
                X.reshape(1, n_features),
                np.ones(n_features).reshape(1, n_features),
            )

            Z = self.cook_test_set(new_X, **kwargs)

            # loop on all the classes
            for i in range(self.n_classes_):
                probs[:, i] = self.fit_objs_[i].predict(Z, **kwargs)[0]

        else:
            Z = self.cook_test_set(X, **kwargs)

            # loop on all the classes
            for i in range(self.n_classes_):
                probs[:, i] = self.fit_objs_[i].predict(Z, **kwargs)

        expit_raw_probs = expit(probs)

        return expit_raw_probs / expit_raw_probs.sum(axis=1)[:, None]
