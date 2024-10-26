# Authors: Thierry Moudiki
#
# License: BSD 3

import numpy as np
import pandas as pd
import sklearn.metrics as skm2
from .custom import Custom
from ..predictionset import PredictionSet
from ..utils import matrixops as mo
from sklearn.base import ClassifierMixin


class CustomClassifier(Custom, ClassifierMixin):
    """Custom Classification model

    Attributes:

        obj: object
            any object containing a method fit (obj.fit()) and a method predict
            (obj.predict())

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
            indicates if the original predictors are included (True) in model''s
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

        level: float
            confidence level for prediction sets. Default is None.

        pi_method: str
            method for constructing the prediction sets: 'icp', 'tcp' if level is not None. Default is 'icp'.

        seed: int
            reproducibility seed for nodes_sim=='uniform'

        backend: str
            "cpu" or "gpu" or "tpu"

    Examples:

    Note: it's better to use the `DeepClassifier` or `LazyDeepClassifier` classes directly

    ```python
    import nnetsauce as ns
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.datasets import load_digits
    from time import time

    digits = load_digits()
    X = digits.data
    y = digits.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        random_state=123)

    # layer 1 (base layer) ----
    layer1_regr = RandomForestClassifier(n_estimators=10, random_state=123)

    start = time()

    layer1_regr.fit(X_train, y_train)

    # Accuracy in layer 1
    print(layer1_regr.score(X_test, y_test))

    # layer 2 using layer 1 ----
    layer2_regr = ns.CustomClassifier(obj = layer1_regr, n_hidden_features=5,
                            direct_link=True, bias=True,
                            nodes_sim='uniform', activation_name='relu',
                            n_clusters=2, seed=123)
    layer2_regr.fit(X_train, y_train)

    # Accuracy in layer 2
    print(layer2_regr.score(X_test, y_test))

    # layer 3 using layer 2 ----
    layer3_regr = ns.CustomClassifier(obj = layer2_regr, n_hidden_features=10,
                            direct_link=True, bias=True, dropout=0.7,
                            nodes_sim='uniform', activation_name='relu',
                            n_clusters=2, seed=123)
    layer3_regr.fit(X_train, y_train)

    # Accuracy in layer 3
    print(layer3_regr.score(X_test, y_test))

    print(f"Elapsed {time() - start}")
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
        level=None,
        pi_method="icp",
        seed=123,
        backend="cpu",
    ):
        super().__init__(
            obj=obj,
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
        self.level = level
        self.pi_method = pi_method
        self.type_fit = "classification"
        if self.level is not None:
            self.obj = PredictionSet(
                self.obj, level=self.level, method=self.pi_method
            )

    def fit(self, X, y, sample_weight=None, **kwargs):
        """Fit custom model to training data (X, y).

        Parameters:

            X: {array-like}, shape = [n_samples, n_features]
                Training vectors, where n_samples is the number
                of samples and n_features is the number of features.

            y: array-like, shape = [n_samples]
                Target values.

            sample_weight: array-like, shape = [n_samples]
                Sample weights.

            **kwargs: additional parameters to be passed to
                        self.cook_training_set or self.obj.fit

        Returns:

            self: object
        """

        if len(X.shape) == 1:
            if isinstance(X, pd.DataFrame):
                X = pd.DataFrame(X.values.reshape(1, -1), columns=X.columns)    
            else:
                X = X.reshape(1, -1)

        output_y, scaled_Z = self.cook_training_set(y=y, X=X, **kwargs)
        self.classes_ =  np.unique(y)
        self.n_classes_ = len(self.classes_)  # for compatibility with         

        if self.level is not None:
            self.obj = PredictionSet(
                obj=self.obj, method=self.pi_method, level=self.level
            )

        # if sample_weights, else: (must use self.row_index)
        if sample_weight is not None:
            self.obj.fit(
                scaled_Z,
                output_y,
                sample_weight=sample_weight[self.index_row_].ravel(),
                # **kwargs
            )

            return self

        # if sample_weight is None:
        self.obj.fit(scaled_Z, output_y)
        self.classes_ = np.unique(y)  # for compatibility with sklearn
        self.n_classes_ = len(self.classes_)  # for compatibility with sklearn

        return self

    def partial_fit(self, X, y, sample_weight=None, **kwargs):
        """Partial fit custom model to training data (X, y).

        Parameters:

            X: {array-like}, shape = [n_samples, n_features]
                Subset of training vectors, where n_samples is the number
                of samples and n_features is the number of features.

            y: array-like, shape = [n_samples]
                Subset of target values.

            sample_weight: array-like, shape = [n_samples]
                Sample weights.

            **kwargs: additional parameters to be passed to
                        self.cook_training_set or self.obj.fit

        Returns:

            self: object
        """

        if len(X.shape) == 1:
            if isinstance(X, pd.DataFrame):
                X = pd.DataFrame(X.values.reshape(1, -1), columns=X.columns)             
            else:
                X = X.reshape(1, -1)
            y = np.array([y], dtype=np.integer)

        output_y, scaled_Z = self.cook_training_set(y=y, X=X, **kwargs)
        self.n_classes_ = len(np.unique(y))  # for compatibility with sklearn

        # if sample_weights, else: (must use self.row_index)
        if sample_weight is not None:
            try:
                self.obj.partial_fit(
                    scaled_Z,
                    output_y,
                    sample_weight=sample_weight[self.index_row_].ravel(),
                    # **kwargs
                )
            except:
                NotImplementedError

            return self

        # if sample_weight is None:
        try:
            self.obj.partial_fit(scaled_Z, output_y)
        except:
            raise NotImplementedError

        self.classes_ = np.unique(y)  # for compatibility with sklearn
        self.n_classes_ = len(self.classes_)  # for compatibility with sklearn

        return self

    def predict(self, X, **kwargs):
        """Predict test data X.

        Parameters:

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
                self.obj.predict(self.cook_test_set(new_X, **kwargs), **kwargs)
            )[0]

        return self.obj.predict(self.cook_test_set(X, **kwargs), **kwargs)

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

            return (
                self.obj.predict_proba(
                    self.cook_test_set(new_X, **kwargs), **kwargs
                )
            )[0]

        return self.obj.predict_proba(self.cook_test_set(X, **kwargs), **kwargs)
