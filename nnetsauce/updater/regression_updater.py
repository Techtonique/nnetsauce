import numpy as np
import pandas as pd
from copy import deepcopy
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted
from ..base import Base
from ..custom import CustomRegressor
from ..deep import DeepRegressor


def _update_mean(mean_old, n_obs, new_vectors):
    return (n_obs * mean_old + new_vectors) / (n_obs + 1)


class RegressorUpdater(BaseEstimator, RegressorMixin):
    """
    Update a regression model with new observations

    Parameters
    ----------
    regr: object
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

    def __init__(self, regr, alpha=0.5):
        self.regr = regr
        self.alpha = alpha
        self.n_obs_ = None
        self.coef_ = None
        self.updating_factor_ = None
        try:
            self.coef_ = self.regr.coef_
            if isinstance(self.regr, Base):
                self.n_obs_ = self.regr.scaler_.n_samples_seen_
        except AttributeError:
            pass

    def fit(self, X, y, **kwargs):

        if isinstance(self.regr, CustomRegressor):  # nnetsauce model not deep ---
            if check_is_fitted(self.regr) == False:
                self.regr.fit(X, y, **kwargs)
                self.n_obs_ = X.shape[0]
                if hasattr(self.regr, "coef_"):
                    self.coef_ = self.regr.coef_
                return self
            self.n_obs_ = self.regr.scaler_.n_samples_seen_
            if hasattr(self.regr, "coef_"):
                self.coef_ = self.regr.coef_
            return self

        if (
            hasattr(self.regr, "coef_") == False
        ):  # sklearn model or CustomRegressor model ---
            self.regr.fit(X, y)
            self.n_obs_ = X.shape[0]
            self.regr.fit(X, y)
            if hasattr(self.regr, "stacked_obj"):
                self.coef_ = self.regr.stacked_obj.coef_
            else:
                self.coef_ = self.regr.coef_
            return self
        self.n_obs_ = X.shape[0]
        if hasattr(self.regr, "coef_"):
            self.coef_ = self.regr.coef_
        return self

    def predict(self, X):
        # assert hasattr(self.regr, "coef_"), "model must have coef_ attribute"
        return self.regr.predict(X)

    def partial_fit(self, X, y):

        assert hasattr(
            self.regr, "coef_"
        ), "model must be fitted first (i.e have 'coef_' attribute)"
        assert (
            self.n_obs_ is not None
        ), "model must be fitted first (i.e have 'n_obs_' attribute)"

        if len(X.shape) == 1:
            X = X.reshape(1, -1)

        assert X.shape[0] == 1, "X must have one row"

        self.updating_factor_ = self.n_obs_ ** (-self.alpha)

        if isinstance(self.regr, Base):  # nnetsauce model ---

            newX = deepcopy(X)

            if isinstance(
                self.regr, CustomRegressor
            ):  # other nnetsauce model (CustomRegressor) ---
                newX = self.regr.cook_test_set(X=X)
                if isinstance(X, pd.DataFrame):
                    newx = newX.values.ravel()
                else:
                    newx = newX.ravel()

        else:  # an sklearn model ---

            if isinstance(X, pd.DataFrame):
                newx = X.values.ravel()
            else:
                newx = X.ravel()

        new_coef = self.regr.coef_ + self.updating_factor_ * np.dot(
            newx, y - np.dot(newx, self.regr.coef_)
        )
        self.regr.coef_ = _update_mean(self.regr.coef_, self.n_obs_, new_coef)
        self.coef_ = deepcopy(self.regr.coef_)
        self.n_obs_ += 1
        return self
