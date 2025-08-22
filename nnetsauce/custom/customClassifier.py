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
from sklearn.calibration import CalibratedClassifierCV


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

        cv_calibration: int, cross-validation generator, or iterable, default=2
            Determines the cross-validation splitting strategy. Same as
            `sklearn.calibration.CalibratedClassifierCV`

        calibration_method: str
            {‘sigmoid’, ‘isotonic’}, default=’sigmoid’
            The method to use for calibration. Same as
            `sklearn.calibration.CalibratedClassifierCV`

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
    _estimator_type = "classifier"

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
        cv_calibration=2,
        calibration_method="sigmoid",
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
        self.coef_ = None
        self.intercept_ = None
        self.type_fit = "classification"
        self.cv_calibration = cv_calibration
        self.calibration_method = calibration_method

    def __sklearn_clone__(self):
        """Create a clone of the estimator.

        This is required for scikit-learn's calibration system to work properly.
        """
        # Create a new instance with the same parameters
        clone = CustomClassifier(
            obj=self.obj,
            n_hidden_features=self.n_hidden_features,
            activation_name=self.activation_name,
            a=self.a,
            nodes_sim=self.nodes_sim,
            bias=self.bias,
            dropout=self.dropout,
            direct_link=self.direct_link,
            n_clusters=self.n_clusters,
            cluster_encode=self.cluster_encode,
            type_clust=self.type_clust,
            type_scaling=self.type_scaling,
            col_sample=self.col_sample,
            row_sample=self.row_sample,
            cv_calibration=self.cv_calibration,
            calibration_method=self.calibration_method,
            seed=self.seed,
            backend=self.backend,
        )
        return clone

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
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)  # for compatibility with sklearn

        # Wrap in CalibratedClassifierCV if needed
        if self.cv_calibration is not None:
            self.obj = CalibratedClassifierCV(
                self.obj, cv=self.cv_calibration, method=self.calibration_method
            )

        # if sample_weights, else: (must use self.row_index)
        if sample_weight is not None:
            self.obj.fit(
                scaled_Z,
                output_y,
                sample_weight=sample_weight[self.index_row_].ravel(),
                **kwargs
            )
            return self

        # if sample_weight is None:
        self.obj.fit(scaled_Z, output_y, **kwargs)
        self.classes_ = np.unique(y)  # for compatibility with sklearn
        self.n_classes_ = len(self.classes_)  # for compatibility with sklearn

        if hasattr(self.obj, "coef_"):
            self.coef_ = self.obj.coef_

        if hasattr(self.obj, "intercept_"):
            self.intercept_ = self.obj.intercept_

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
        # try:
        self.obj.partial_fit(scaled_Z, output_y)
        # except:
        #    raise NotImplementedError

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

    def decision_function(self, X, **kwargs):
        """Compute the decision function of X.

        Parameters:
            X: {array-like}, shape = [n_samples, n_features]
                Samples to compute decision function for.

            **kwargs: additional parameters to be passed to
                    self.cook_test_set

        Returns:
            array-like of shape (n_samples,) or (n_samples, n_classes)
            Decision function of the input samples. The order of outputs is the same
            as that of the classes passed to fit.
        """
        if not hasattr(self.obj, "decision_function"):
            # If base classifier doesn't have decision_function, use predict_proba
            proba = self.predict_proba(X, **kwargs)
            if proba.shape[1] == 2:
                return proba[:, 1]  # For binary classification
            return proba  # For multiclass

        if len(X.shape) == 1:
            n_features = X.shape[0]
            new_X = mo.rbind(
                X.reshape(1, n_features),
                np.ones(n_features).reshape(1, n_features),
            )

            return (
                self.obj.decision_function(
                    self.cook_test_set(new_X, **kwargs), **kwargs
                )
            )[0]

        return self.obj.decision_function(
            self.cook_test_set(X, **kwargs), **kwargs
        )

    def score(self, X, y, scoring=None):
        """Scoring function for classification.

        Args:

            X: {array-like}, shape = [n_samples, n_features]
                Training vectors, where n_samples is the number
                of samples and n_features is the number of features.

            y: array-like, shape = [n_samples]
                Target values.

            scoring: str
                scoring method (default is accuracy)

        Returns:

            score: float
        """

        if scoring is None:
            scoring = "accuracy"

        if scoring == "accuracy":
            return skm2.accuracy_score(y, self.predict(X))

        if scoring == "f1":
            return skm2.f1_score(y, self.predict(X))

        if scoring == "precision":
            return skm2.precision_score(y, self.predict(X))

        if scoring == "recall":
            return skm2.recall_score(y, self.predict(X))

        if scoring == "roc_auc":
            return skm2.roc_auc_score(y, self.predict(X))

        if scoring == "log_loss":
            return skm2.log_loss(y, self.predict_proba(X))

        if scoring == "balanced_accuracy":
            return skm2.balanced_accuracy_score(y, self.predict(X))

        if scoring == "average_precision":
            return skm2.average_precision_score(y, self.predict(X))

        if scoring == "neg_brier_score":
            return -skm2.brier_score_loss(y, self.predict_proba(X))

        if scoring == "neg_log_loss":
            return -skm2.log_loss(y, self.predict_proba(X))

    @property
    def _estimator_type(self):
        return "classifier"
