# Authors: Thierry Moudiki
#
# License: BSD 3

import numpy as np
import sklearn.metrics as skm2
from ..base import Base
from ..utils import matrixops as mo
from ..utils import misc as mx
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

        self.scaled_X_ = self.X_scaler_.fit_transform(X)

        self.n_classes_ = len(np.unique(y))

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

            # loop on all the classes
            for i in range(self.n_classes_):
                probs[:, i] = self.fit_objs_[i].predict(Z, **kwargs)[0]

        else:
            Z = self.X_scaler_.transform(X, **kwargs)

            # loop on all the classes
            for i in range(self.n_classes_):
                probs[:, i] = self.fit_objs_[i].predict(Z, **kwargs)

        expit_raw_probs = expit(probs)

        return expit_raw_probs / expit_raw_probs.sum(axis=1)[:, None]

    def score(self, X, y, scoring=None, **kwargs):
        """ Score the model on test set features X and response y. 

        Args:

            X: {array-like}, shape = [n_samples, n_features]
                Training vectors, where n_samples is the number 
                of samples and n_features is the number of features

            y: array-like, shape = [n_samples]
                Target values

            scoring: str
                must be in ('explained_variance', 'neg_mean_absolute_error', \
                            'neg_mean_squared_error', 'neg_mean_squared_log_error', \
                            'neg_median_absolute_error', 'r2')

            **kwargs: additional parameters to be passed to scoring functions

        Returns: 

            model scores: {array-like}

        """

        preds = self.predict(X)

        if scoring is None:
            scoring = "accuracy"

        # check inputs
        assert scoring in (
            "accuracy",
            "average_precision",
            "brier_score_loss",
            "f1",
            "f1_micro",
            "f1_macro",
            "f1_weighted",
            "f1_samples",
            "neg_log_loss",
            "precision",
            "recall",
            "roc_auc",
        ), "'scoring' should be in ('accuracy', 'average_precision', \
                           'brier_score_loss', 'f1', 'f1_micro', \
                           'f1_macro', 'f1_weighted',  'f1_samples', \
                           'neg_log_loss', 'precision', 'recall', \
                           'roc_auc')"

        scoring_options = {
            "accuracy": skm2.accuracy_score,
            "average_precision": skm2.average_precision_score,
            "brier_score_loss": skm2.brier_score_loss,
            "f1": skm2.f1_score,
            "f1_micro": skm2.f1_score,
            "f1_macro": skm2.f1_score,
            "f1_weighted": skm2.f1_score,
            "f1_samples": skm2.f1_score,
            "neg_log_loss": skm2.log_loss,
            "precision": skm2.precision_score,
            "recall": skm2.recall_score,
            "roc_auc": skm2.roc_auc_score,
        }

        return scoring_options[scoring](y, preds, **kwargs)
