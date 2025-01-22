import numpy as np
import pandas as pd
from copy import deepcopy
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted
from ..base import Base
from ..custom import CustomClassifier
from ..deep import DeepClassifier


def _update_mean(mean_old, n_obs, new_vectors):
    return (n_obs * mean_old + new_vectors) / (n_obs + 1)


class ClassifierUpdater(BaseEstimator, ClassifierMixin):
    """
    Update a regression model with new observations

    Parameters
    ----------
    clf: object
        A regression model with a coef_ attribute
    alpha: float
        Updating factor's exponent

    Attributes
    ----------
    n_obs_: int
        Number of observations
    coef_: np.ndarray
        Coefficients of the model
    updating_factor_: float
        Updating factor

    """

    def __init__(self, clf, alpha=0.5):
        self.clf = clf
        self.alpha = alpha
        self.n_obs_ = None
        self.coef_ = None
        self.updating_factor_ = None
        try:
            self.coef_ = self.clf.coef_
            if isinstance(self.clf, Base):
                self.n_obs_ = self.clf.scaler_.n_samples_seen_
        except AttributeError:
            pass

    def fit(self, X, y, **kwargs):

        raise NotImplementedError("fit method is not implemented for ClassifierUpdater")

        if isinstance(self.clf, CustomClassifier):  # nnetsauce model not deep ---
            if check_is_fitted(self.clf) == False:
                self.clf.fit(X, y, **kwargs)
                self.n_obs_ = X.shape[0]
                if hasattr(self.clf, "coef_"):
                    self.coef_ = self.clf.coef_
                return self
            self.n_obs_ = self.clf.scaler_.n_samples_seen_
            if hasattr(self.clf, "coef_"):
                self.coef_ = self.clf.coef_
            return self

        if (
            hasattr(self.clf, "coef_") == False
        ):  # sklearn model or CustomClassifier model ---
            self.clf.fit(X, y)
            self.n_obs_ = X.shape[0]
            self.clf.fit(X, y)
            if hasattr(self.clf, "stacked_obj"):
                self.coef_ = self.clf.stacked_obj.coef_
            else:
                self.coef_ = self.clf.coef_
            return self
        self.n_obs_ = X.shape[0]
        if hasattr(self.clf, "coef_"):
            self.coef_ = self.clf.coef_
        return self

    def predict(self, X):

        raise NotImplementedError(
            "predict method is not implemented for ClassifierUpdater"
        )
        # assert hasattr(self.clf, "coef_"), "model must have coef_ attribute"
        return self.clf.predict(X)

    def partial_fit(self, X, y):

        raise NotImplementedError(
            "partial_fit method is not implemented for ClassifierUpdater"
        )

        assert hasattr(
            self.clf, "coef_"
        ), "model must be fitted first (i.e have 'coef_' attribute)"
        assert (
            self.n_obs_ is not None
        ), "model must be fitted first (i.e have 'n_obs_' attribute)"

        if len(X.shape) == 1:
            X = X.reshape(1, -1)

        assert X.shape[0] == 1, "X must have one row"

        self.updating_factor_ = self.n_obs_ ** (-self.alpha)

        if isinstance(self.clf, Base):  # nnetsauce model ---

            newX = deepcopy(X)

            if isinstance(
                self.clf, CustomClassifier
            ):  # other nnetsauce model (CustomClassifier) ---
                newX = self.clf.cook_test_set(X=X)
                if isinstance(X, pd.DataFrame):
                    newx = newX.values.ravel()
                else:
                    newx = newX.ravel()

        else:  # an sklearn model ---

            if isinstance(X, pd.DataFrame):
                newx = X.values.ravel()
            else:
                newx = X.ravel()

        new_coef = self.clf.coef_ + self.updating_factor_ * np.dot(
            newx, y - np.dot(newx, self.clf.coef_)
        )
        self.clf.coef_ = _update_mean(self.clf.coef_, self.n_obs_, new_coef)
        self.coef_ = deepcopy(self.clf.coef_)
        self.n_obs_ += 1
        return self
