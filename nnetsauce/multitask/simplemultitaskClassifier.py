# Authors: Thierry Moudiki
#
# License: BSD 3

import numpy as np
import sklearn.metrics as skm2
from ..base import Base
from ..utils import matrixops as mo
from ..utils import misc as mx
from collections import namedtuple
from copy import deepcopy
from scipy.special import expit
from sklearn.base import ClassifierMixin
from sklearn.preprocessing import StandardScaler


class SimpleMultitaskClassifier(Base, ClassifierMixin):
    """Multitask Classification model based on regression models, with shared covariates

    Parameters:

        obj: object
            any object (must be a regression model) containing a method fit (obj.fit())
            and a method predict (obj.predict())

        seed: int
            reproducibility seed

    Attributes:

        fit_objs_: dict
            objects adjusted to each individual time series

        n_classes_: int
            number of classes for the classifier

    Examples:

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
    fit_obj = ns.SimpleMultitaskClassifier(regr)

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
    _estimator_type = "classifier"

    def __init__(
        self,
        obj,
    ):
        self.type_fit = "classification"
        self.obj = obj
        self.fit_objs_ = {}
        self.X_scaler_ = StandardScaler()
        self.scaled_X_ = None

    def fit(self, X, y, sample_weight=None, **kwargs):
        """Fit SimpleMultitaskClassifier to training data (X, y).

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

        self.classes_ = np.unique(y)  # for compatibility with sklearn
        self.n_classes_ = len(self.classes_)  # for compatibility with sklearn

        self.scaled_X_ = self.X_scaler_.fit_transform(X)

        # multitask response
        Y = mo.one_hot_encode2(y, self.n_classes_)

        try: 
            for i in range(self.n_classes_):
                self.fit_objs_[i] = deepcopy(
                    self.obj.fit(self.scaled_X_, Y[:, i], sample_weight=sample_weight, **kwargs)
                )
        except Exception as e:
            for i in range(self.n_classes_):
                self.fit_objs_[i] = deepcopy(
                    self.obj.fit(self.scaled_X_, Y[:, i], **kwargs)
                )
        return self

    def predict(self, X, **kwargs):
        """Predict test data X.

        Args:

            X: {array-like}, shape = [n_samples, n_features]
                Training vectors, where n_samples is the number
                of samples and n_features is the number of features.

            **kwargs: additional parameters

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

            **kwargs: additional parameters

        Returns:

            probability estimates for test data: {array-like}

        """

        shape_X = X.shape

        probs = np.zeros((shape_X[0], self.n_classes_))

        if len(shape_X) == 1: # one example

            n_features = shape_X[0]

            new_X = mo.rbind(
                X.reshape(1, n_features),
                np.ones(n_features).reshape(1, n_features),
            )

            Z = self.X_scaler_.transform(new_X, **kwargs)

            # Fallback to standard model
            for i in range(self.n_classes_):
                probs[:, i] = self.fit_objs_[i].predict(Z, **kwargs)[0]

        else: # multiple rows

            Z = self.X_scaler_.transform(X, **kwargs)

            # Fallback to standard model
            for i in range(self.n_classes_):
                probs[:, i] = self.fit_objs_[i].predict(Z, **kwargs)

        expit_raw_probs = expit(probs)
        
        # Add small epsilon to avoid division by zero
        row_sums = expit_raw_probs.sum(axis=1)[:, None]
        row_sums[row_sums < 1e-10] = 1e-10
        
        return expit_raw_probs / row_sums

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

        return self.obj.decision_function(self.cook_test_set(X, **kwargs), **kwargs)

    @property
    def _estimator_type(self):
        return "classifier"            
