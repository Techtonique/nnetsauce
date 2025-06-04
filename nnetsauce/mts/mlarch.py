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
        self.mean_residuals_kss_test_ = None
        self.standardized_residuals_ = None


    def fit(self, y):
        n = len(y)
        self.model_mean.fit(y.reshape(-1, 1)) 
        # Wilcoxon signed-rank test on residuals (mean = 0)
        self.mean_residuals_wilcoxon_test_ = stats.wilcoxon(self.model_mean.residuals_)
        # KPSS test for stationarity on residuals
        self.mean_residuals_kss_test_ = kpss(self.model_mean.residuals_, regression='c')
        self.model_sigma.fit(np.log(self.model_mean.residuals_.reshape(-1, 1)**2)) 
        # n//2 here because the model is conformalized
        fitted_sigma = self.model_sigma.residuals_ + np.log(self.model_mean.residuals_**2)[(n//2):,:]
        # standardized residuals
        self.standardized_residuals_ = self.model_mean.residuals_[(n//2):,:]/np.sqrt(np.exp(fitted_sigma))
        self.model_residuals.fit(self.standardized_residuals_.reshape(-1, 1))
        return self


    def predict(self, h=5, level=95):
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



