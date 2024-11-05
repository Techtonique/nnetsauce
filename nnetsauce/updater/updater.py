import numpy as np
import pandas as pd
from copy import deepcopy
from sklearn.base import BaseEstimator, RegressorMixin


def _update_mean(mean_old, n_obs, new_vectors):
    print(f"n_obs: {n_obs}")
    return (n_obs * mean_old + new_vectors) / (n_obs + 1)

class RegressorUpdater(BaseEstimator, RegressorMixin):
    def __init__(self, regr, alpha=0.5):
        self.regr = regr
        self.alpha = alpha
        self.n_obs_ = None
        self.coef_ = None 
        self.updating_factor_ = None        

    def fit(self, X, y):
        self.n_obs_ = X.shape[0]
        self.regr.fit(X, y)  
        self.coef_ = self.regr.coef_      
        return self

    def predict(self, X):
        assert hasattr(self.regr, "coef_"), "model must have coef_ attribute"
        return self.regr.predict(X) 

    def partial_fit(self, X, y):        
        assert hasattr(self.regr, "coef_"), "model must be fitted first (must have coef_ attribute)" 
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        assert X.shape[0] == 1, "X must have 1 row"       
        self.updating_factor_ = self.n_obs_**(-self.alpha)
        if isinstance(X, pd.DataFrame):
            new_coef = self.regr.coef_ + self.updating_factor_ * np.dot(X.values.ravel(), y - np.dot(X.values.ravel(), 
                                                                                                 self.regr.coef_))
        else: 
            new_coef = self.regr.coef_ + self.updating_factor_ * np.dot(X.ravel(), y - np.dot(X.ravel(), self.regr.coef_))       
        print(f"new_coef: {new_coef}")    
        self.regr.coef_ = _update_mean(self.regr.coef_, self.n_obs_, new_coef)
        self.coef_ = deepcopy(self.regr.coef_)
        self.n_obs_ += X.shape[0]
        return self

