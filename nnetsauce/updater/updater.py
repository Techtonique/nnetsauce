import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin


class RegressorUpdater(BaseEstimator, RegressorMixin):
    def __init__(self, regr, alpha=0.5):
        self.regr = regr
        self.alpha = alpha

    def fit(self, X, y):
        self.n_obs = X.shape[0]
        self.regr.fit(X, y)        
        return self.regr 

    def predict(self, X):
        assert hasattr(self.regr, "coef_"), "model must have coef_ attribute"
        return self.regr.predict(X) 

    def partial_fit(self, X, y):
        self.n_obs += X.shape[0]
        self.updating_factor_ = self.n_obs**(-self.alpha)
        assert hasattr(self.regr, "coef_"), "model must have coef_ attribute"
        
        return self.regr

