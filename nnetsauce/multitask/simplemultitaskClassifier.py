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

        # if sample_weight is None:
        for i in range(self.n_classes_):
            self.fit_objs_[i] = deepcopy(
                self.obj.fit(self.scaled_X_, Y[:, i], **kwargs)
            )
        self.classes_ = np.unique(y)
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
        try:
            preds = self.predict_proba(X, **kwargs)
            try:
                DescribeResult = namedtuple(
                    "DescribeResult", ["mean", "upper", "lower", "median"]
                )
                return DescribeResult(
                    mean=np.argmax(preds.mean, axis=1),
                    upper=np.argmax(preds.upper, axis=1),
                    lower=np.argmax(preds.lower, axis=1),
                    median=np.argmax(preds.median, axis=1),
                )
            except Exception as e:

                DescribeResult = namedtuple(
                    "DescribeResult", ["mean", "upper", "lower"]
                )
                return DescribeResult(
                    mean=np.argmax(preds.mean, axis=1),
                    upper=np.argmax(preds.upper, axis=1),
                    lower=np.argmax(preds.lower, axis=1),
                )
        except Exception as e:

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

        if len(shape_X) == 1:
            n_features = shape_X[0]

            new_X = mo.rbind(
                X.reshape(1, n_features),
                np.ones(n_features).reshape(1, n_features),
            )

            Z = self.X_scaler_.transform(new_X, **kwargs)

            try:
                # Try probabilistic model first (conformal or quantile)
                probs_upper = np.zeros((shape_X[0], self.n_classes_))
                probs_lower = np.zeros((shape_X[0], self.n_classes_))
                probs_median = np.zeros((shape_X[0], self.n_classes_))

                # loop on all the classes
                for i in range(self.n_classes_):
                    probs_temp = self.fit_objs_[i].predict(Z, **kwargs)
                    probs_upper[:, i] = probs_temp.upper
                    probs_lower[:, i] = probs_temp.lower
                    probs[:, i] = probs_temp.mean
                    try:
                        probs_median[:, i] = probs_temp.median
                    except:
                        pass

            except Exception as e:

                # Fallback to standard model
                for i in range(self.n_classes_):
                    probs[:, i] = self.fit_objs_[i].predict(Z, **kwargs)[0]

        else:

            Z = self.X_scaler_.transform(X, **kwargs)

            try:
                # Try probabilistic model first (conformal or quantile)
                probs_upper = np.zeros((shape_X[0], self.n_classes_))
                probs_lower = np.zeros((shape_X[0], self.n_classes_))
                probs_median = np.zeros((shape_X[0], self.n_classes_))

                # loop on all the classes
                for i in range(self.n_classes_):
                    probs_temp = self.fit_objs_[i].predict(Z, **kwargs)
                    probs_upper[:, i] = probs_temp.upper
                    probs_lower[:, i] = probs_temp.lower
                    probs[:, i] = probs_temp.mean
                    try:
                        probs_median[:, i] = probs_temp.median
                    except:
                        pass

            except Exception as e:

                # Fallback to standard model
                for i in range(self.n_classes_):
                    probs[:, i] = self.fit_objs_[i].predict(Z, **kwargs)[0]

        expit_raw_probs = expit(probs)

        try:
            expit_raw_probs_upper = expit(probs_upper)
            expit_raw_probs_lower = expit(probs_lower)
            try:
                expit_raw_probs_median = expit(probs_median)
            except Exception as e:

                pass
            probs_upper = (
                expit_raw_probs_upper / expit_raw_probs_upper.sum(axis=1)[:, None]
            )
            probs_lower = (
                expit_raw_probs_lower / expit_raw_probs_lower.sum(axis=1)[:, None]
            )
            probs_upper = np.minimum(probs_upper, 1)
            probs_lower = np.maximum(probs_lower, 0)
            try:
                probs_median = (
                    expit_raw_probs_median / expit_raw_probs_median.sum(axis=1)[:, None]
                )
            except Exception as e:

                pass

            # Normalize each probability independently to [0,1] range
            probs = expit_raw_probs
            probs_upper = np.minimum(expit_raw_probs_upper, 1)
            probs_lower = np.maximum(expit_raw_probs_lower, 0)

            # Ensure upper >= lower
            probs_upper = np.maximum(probs_upper, probs_lower)

            try:
                probs_median = expit_raw_probs_median
            except Exception as e:

                pass

            try:
                DescribeResult = namedtuple(
                    "DescribeResult", ["mean", "upper", "lower", "median"]
                )
                return DescribeResult(
                    mean=probs,
                    upper=probs_upper,
                    lower=probs_lower,
                    median=probs_median,
                )
            except Exception as e:

                DescribeResult = namedtuple(
                    "DescribeResult", ["mean", "upper", "lower"]
                )
                return DescribeResult(mean=probs, upper=probs_upper, lower=probs_lower)

        except Exception as e:

            return expit_raw_probs / expit_raw_probs.sum(axis=1)[:, None]
