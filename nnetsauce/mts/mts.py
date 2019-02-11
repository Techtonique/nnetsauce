"""MTS model"""

# Authors: Thierry Moudiki <thierry.moudiki@gmail.com>
#
# License: MIT

# case with confidence intervals on predictions for obj (ex. Gaussian process)
# case with confidence intervals on predictions for obj (ex. Gaussian process)
# case with confidence intervals on predictions for obj (ex. Gaussian process)
# case with confidence intervals on predictions for obj (ex. Gaussian process)
# case with confidence intervals on predictions for obj (ex. Gaussian process)

# PCA 
# PCA 
# PCA 
# PCA 
# PCA 

import numpy as np
from ..base import Base
from ..utils import matrixops as mo
from ..utils import misc as mx
from ..utils import timeseries as ts


class MTS(Base):
    """MTS model class derived from class Base
    
       Parameters
       ----------
       obj: object
           any object containing a method fit (obj.fit()) and a method predict 
           (obj.predict())
       n_hidden_features: int
           number of nodes in the hidden layer
       activation_name: str
           activation function: 'relu', 'tanh', 'sigmoid', 'prelu' or 'elu'
       a: float
           hyperparameter for 'prelu' or 'elu' activation function
       nodes_sim: str
           type of simulation for the nodes: 'sobol', 'hammersley', 'halton', 
           'uniform'
       bias: boolean
           indicates if the hidden layer contains a bias term (True) or not 
           (False)
       direct_link: boolean
           indicates if the original predictors are included (True) in model's 
           fitting or not (False)
       n_clusters: int
           number of clusters for 'kmeans' or 'gmm' clustering (could be 0: 
               no clustering)
       type_clust: str
           type of clustering method: currently k-means ('kmeans') or Gaussian 
           Mixture Model ('gmm')
       seed: int 
           reproducibility seed for nodes_sim=='uniform'
       lags: int
           number of lags for the time series 
       fit_objs: dict
           a dictionary containing objects fitted with obj, one for each time series
    """
    
    # construct the object -----
        
    def __init__(self, obj, n_hidden_features=5, 
                 activation_name='relu', a=0.01,
                 nodes_sim='sobol', bias=True,
                 direct_link=True, n_clusters=2,
                 type_clust='kmeans', seed=123, 
                 lags = 1):
        
        assert np.int(lags) == lags, "parameter 'lags' should be an integer"        
        
        super().__init__(n_hidden_features = n_hidden_features, 
                         activation_name = activation_name, a = a,
                         nodes_sim = nodes_sim, bias = bias, 
                         direct_link = direct_link,
                         n_clusters = n_clusters, 
                         type_clust = type_clust, 
                         seed = seed)
        
        self.obj = obj
        self.n_series = None
        self.lags = lags
        self.fit_objs = {}
        self.y = None # MTS responses (most recent observations first)
        self.X = None # MTS lags
        self.y_means = []
        self.preds = None
        self.preds_std = []


    def get_params(self):
        
        return mx.merge_two_dicts(super().get_params(), 
                                  {'n_series': self.n_series, 
                                   'lags': self.lags})

    
    def set_params(self, n_hidden_features=5, 
                   activation_name='relu', a=0.01,
                   nodes_sim='sobol', bias=True,
                   direct_link=True, n_clusters=None,
                   type_clust='kmeans', seed=123, 
                   lags = 1):
        
        super().set_params(n_hidden_features = n_hidden_features, 
                           activation_name = activation_name, a = a,
                           nodes_sim = nodes_sim, bias = bias, 
                           direct_link = direct_link, n_clusters = n_clusters, 
                           type_clust = type_clust, seed = seed)
        self.lags = lags
 
    
    def fit(self, X, **kwargs):
        """Fit MTS model to training data (X, y).
        
        Parameters
        ----------
        X: {array-like}, shape = [n_samples, n_features]
            Training time series, where n_samples is the number 
            of samples and n_features is the number of features.
            X must be in decreasing order (most recent observations first)
        
        **kwargs: additional parameters to be passed to 
                  self.cook_training_set
               
        Returns
        -------
        self: object
        """
        
        self.y = None
        
        self.X = None
        
        n, p = X.shape
        
        self.n_series = p
        
        self.y_means = np.zeros(p)
        
        mts_input = ts.create_train_inputs(X, self.lags)
        
        self.y = mts_input[0]
        
        self.X = mts_input[1]
        
        # avoids scaling X two times in the loop
        scaled_Z = self.cook_training_set(y = np.repeat(1, self.n_series), 
                                          X = self.X, **kwargs)
        
        # loop on all the time series and adjust self.obj.fit
        for i in range(p):
            y = self.y[:,i]
            y_mean = np.mean(y)
            self.y_mean = y_mean
            centered_y = y - y_mean
            self.y_means[i] = y_mean
            self.y_mean = None
            self.fit_objs[i] = self.obj.fit(scaled_Z, centered_y, **kwargs)
        
        return self

    
    def predict(self, h = 5, **kwargs):
        """Predict on horizon h.
        
        Parameters
        ----------
        h: {integer}
            Forecasting horizon
        
        **kwargs: additional parameters to be passed to 
                  self.cook_test_set
               
        Returns
        -------
        model predictions for horizon = h: {array-like}
        """
        
        self.preds = None
        
        self.preds = self.y
        
        n_features = self.n_series*self.lags
        
        self.preds_std = np.zeros(h)
        
        # change this: put loop inside of if
        # change this: put loop inside of if
        # change this: put loop inside of if
        # change this: put loop inside of if
        # change this: put loop inside of if
        
        for i in range(h):
        
            new_obs = ts.reformat_response(self.preds, self.lags)
            
            new_X = mo.rbind(new_obs.reshape(1, n_features), 
                             np.ones(n_features).reshape(1, n_features))        
                
            cooked_new_X = self.cook_test_set(new_X, **kwargs)
            
            predicted_cooked_new_X = self.obj.predict(cooked_new_X, 
                                                      **kwargs)
            
            if isinstance(predicted_cooked_new_X, tuple) == False: # the std. dev. is returned
                
                preds = np.array([(self.y_means[j] + predicted_cooked_new_X[0]) for j in range(self.n_series)])
                
                self.preds = np.row_stack((preds, self.preds))
                
            else:
                
                preds = np.array([(self.y_means[j] + predicted_cooked_new_X[0][0]) for j in range(self.n_series)])
                
                self.preds = np.row_stack((preds, self.preds))
                
                self.preds_std[i] = predicted_cooked_new_X[1][0]
        
        res_preds = self.preds[0:h,:]
        
        if isinstance(predicted_cooked_new_X, tuple) == False: # the std. dev. is returned
            return(res_preds)
        else:
            return(res_preds, 
                   res_preds - self.preds_std.reshape(h, 1), 
                   res_preds + self.preds_std.reshape(h, 1))
            
                
        