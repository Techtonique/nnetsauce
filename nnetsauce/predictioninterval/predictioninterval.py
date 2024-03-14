from locale import normalize
import numpy as np
import pickle
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesRegressor
from ..nonconformist import IcpRegressor
from ..nonconformist import RegressorNc 
from ..nonconformist import RegressorNormalizer, AbsErrorErrFunc
from ..utils import Progbar


class PredictionInterval(BaseEstimator, RegressorMixin):
    """Class PredictionInterval: Obtain prediction intervals.
        
    Attributes:
       
        obj: an object;
            fitted object containing methods `fit` and `predict`

        method: a string;
            method for constructing the prediction intervals. 
            Currently "splitconformal" (default) and "localconformal"

        level: a float;                
            Confidence level for prediction intervals. Default is 0.95, 
            equivalent to a miscoverage error of 0.05
        
        seed: an integer;
            Reproducibility of fit (there's a random split between fitting and calibration data)
    """

    def __init__(self, obj, method="splitconformal", level=0.95, seed=123):

        self.obj = obj
        self.method = method
        self.level = level
        self.seed = seed
        self.quantile_ = None
        self.icp_ = None


    def fit(self, X, y):
        """Fit the `method` to training data (X, y).           
        
        Args:

            X: array-like, shape = [n_samples, n_features]; 
                Training set vectors, where n_samples is the number 
                of samples and n_features is the number of features.                

            y: array-like, shape = [n_samples, ]; Target values.
                       
        """       

        X_train, X_calibration, y_train, y_calibration = train_test_split(X, y, 
                                                    test_size=0.5, random_state=self.seed)

        if self.method == "splitconformal": 

            n_samples_calibration = X_calibration.shape[0]
            q = self.level*(1 + 1/n_samples_calibration) 
            self.obj.fit(X_train, y_train)
            preds_calibration = self.obj.predict(X_calibration)
            absolute_residuals = np.abs(y_calibration - preds_calibration)         
            try: 
                # numpy version >= 1.22
                self.quantile_ = np.quantile(a = absolute_residuals, q = q, 
                method="higher")                                       
            except:
                # numpy version < 1.22
                self.quantile_ = np.quantile(a = absolute_residuals, q = q, 
                interpolation="higher")           
                

        if self.method == "localconformal":

            mad_estimator = ExtraTreesRegressor()
            normalizer = RegressorNormalizer(self.obj, mad_estimator, AbsErrorErrFunc())
            nc = RegressorNc(self.obj, AbsErrorErrFunc(), normalizer)
        
            self.icp_ = IcpRegressor(nc)
            self.icp_.fit(X_train, y_train) 
            self.icp_.calibrate(X_calibration, y_calibration)

        return self


    def predict(self, X, return_pi=False):
        """Obtain predictions and prediction intervals            

        Args: 

            X: array-like, shape = [n_samples, n_features]; 
                Testing set vectors, where n_samples is the number 
                of samples and n_features is the number of features. 

            return_pi: boolean               
                Whether the prediction interval is returned or not. 
                Default is False, for compatibility with other _estimators_.
                If True, a tuple containing the predictions + lower and upper 
                bounds is returned.

        """                

        pred = self.obj.predict(X)

        if self.method == "splitconformal":

            if return_pi:
                
                return pred, (pred - self.quantile_), (pred + self.quantile_)

            else: 

                return pred

        if self.method == "localconformal":

            if return_pi:

                predictions_bounds = self.icp_.predict(X, significance = 1-self.level)
                return pred, predictions_bounds[:, 0], predictions_bounds[:, 1]

            else:
                
                return pred

