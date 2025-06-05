# Authors: T. Moudiki
#
# License: BSD 3 Clear Clause

import numpy as np
import pandas as pd
from collections import namedtuple
from scipy.stats import norm
from scipy import stats
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss
from ..base import Base
from .mts import MTS
from ..utils import timeseries as ts


class MLARCH(MTS):
    """Machine Learning with ARCH effects for time series forecasting

    Parameters:
            
        model_mean: object of class nnetsauce.MTS
            Model for mean prediction (default: None, uses obj)
            
        model_sigma: object of class nnetsauce.MTS
            Model for residuals volatility prediction (default: None, uses obj)
        
        model_residuals: object of class nnetsauce.MTS
            Model for residuals prediction (default: None, uses obj)
    
    Examples: 

        See examples/mlarch.py
                        
    """
    def __init__(
        self,
        model_mean,
        model_sigma, 
        model_residuals
    ):
        assert isinstance(model_mean, MTS), "model_mean must be an object of class nnetsauce.MTS"
        assert isinstance(model_sigma, MTS), "model_sigma must be an object of class nnetsauce.MTS"
        assert isinstance(model_residuals, MTS), "model_residuals must be an object of class nnetsauce.MTS"
        assert model_sigma.type_pi.startswith("scp") and model_sigma.replications is not None, \
        "for now, the models must be conformalized, i.e type_pi must start with 'scp' and replications must be an integer"
        assert model_residuals.type_pi.startswith("scp") and model_residuals.replications is not None, \
        "for now, the models must be conformalized, i.e type_pi must start with 'scp' and replications must be an integer"        

        self.model_mean = model_mean
        self.model_sigma = model_sigma
        self.model_residuals = model_residuals

        self.mean_residuals_ = None
        self.mean_residuals_wilcoxon_test_ = None
        self.mean_residuals_kpss_test_ = None
        self.standardized_residuals_ = None


    def fit(self, y):
        """Fit the MLARCH model to the time series data.

        Parameters
        ----------
        y : array-like of shape (n_samples,)
            The target time series to be fitted.

        Returns
        -------
        self : object
            Returns self.

        Notes
        -----
        This method:

        1. Fits the mean model to the time series
        2. Performs statistical tests on the residuals (Wilcoxon and KPSS)
        3. Fits the volatility model to the squared residuals
        4. Computes standardized residuals
        5. Fits the residuals model to the standardized residuals
        """
        n = len(y)
        self.model_mean.fit(y.reshape(-1, 1)) 
        # Wilcoxon signed-rank test on residuals (mean = 0)
        self.mean_residuals_wilcoxon_test_ = stats.wilcoxon(self.model_mean.residuals_)
        # KPSS test for stationarity on residuals
        self.mean_residuals_kpss_test_ = kpss(self.model_mean.residuals_, regression='c')
        self.model_sigma.fit(np.log(self.model_mean.residuals_.reshape(-1, 1)**2)) 
        # n//2 here because the model is conformalized
        fitted_sigma = self.model_sigma.residuals_ + np.log(self.model_mean.residuals_**2)[(n//2):,:]
        # standardized residuals
        self.standardized_residuals_ = self.model_mean.residuals_[(n//2):,:]/np.sqrt(np.exp(fitted_sigma))
        self.model_residuals.fit(self.standardized_residuals_.reshape(-1, 1))

        # Calculate AIC
        # Get predictions from all models
        mean_pred = self.model_mean.predict(h=0).values.ravel()
        sigma_pred = self.model_sigma.predict(h=0).values.ravel()
        z_pred = self.model_residuals.predict(h=0).values.ravel()
        
        # Calculate combined predictions
        combined_pred = mean_pred + z_pred * np.sqrt(np.exp(sigma_pred))
        
        # Calculate SSE using the last half of the data (matching standardized_residuals_)
        y_actual = y[(n//2):].ravel()
        self.sse_ = np.sum((y_actual - combined_pred) ** 2)
        
        # Calculate number of parameters (sum of parameters from all three models)
        n_params = (self.model_mean.n_hidden_features + 1 +  # mean model
                   self.model_sigma.n_hidden_features + 1 +  # sigma model
                   self.model_residuals.n_hidden_features + 1)  # residuals model
        
        # Calculate AIC
        n_samples = len(y_actual)
        self.aic_ = n_samples * np.log(self.sse_/n_samples) + 2 * n_params

        return self


    def predict(self, h=5, level=95):
        """Predict (probabilistic) future values of the time series.

        Parameters
        ----------
        h : int, default=5
            The forecast horizon.
        level : int, default=95
            The confidence level for prediction intervals.

        Returns
        -------
        DescribeResult : namedtuple
            A named tuple containing:

            - mean : array-like of shape (h,)
                The mean forecast.
            - sims : array-like of shape (h, n_replications)
                The simulated forecasts.
            - lower : array-like of shape (h,)
                The lower bound of the prediction interval.
            - upper : array-like of shape (h,)
                The upper bound of the prediction interval.

        Notes
        -----
        This method:
        1. Generates mean forecasts using the mean model
        2. Generates standardized residual forecasts using the residuals model
        3. Generates volatility forecasts using the sigma model
        4. Combines these forecasts to generate the final predictions
        5. Computes prediction intervals at the specified confidence level
        """
        DescribeResult = namedtuple(
                "DescribeResult", ("mean", "sims", "lower", "upper")
            )
        mean_forecast = self.model_mean.predict(h=h).values.ravel()
        preds_z = self.model_residuals.predict(h=h)
        preds_sigma = self.model_sigma.predict(h=h)
        sims_z = preds_z.sims
        sims_sigma = preds_sigma.sims 

        f = []
        for i in range(len(sims_z)): 
            f.append(mean_forecast + sims_z[i].values.ravel()*np.sqrt(np.exp(sims_sigma[i].values.ravel())))

        f = np.asarray(f).T
        mean_f = np.mean(f, axis=1)
        alpha = 1 - level/100
        lower_bound = np.quantile(f, alpha/2, axis=1)
        upper_bound = np.quantile(f, 1-alpha/2, axis=1)

        return DescribeResult(mean_f, f, 
                              lower_bound, upper_bound)



