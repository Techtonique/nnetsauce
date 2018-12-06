import numpy as np
from ..base import Base
from ..utils import misc as mx
from ..utils import matrixops as mo
from ..utils import lmfuncs as lmf


class BayesianRVFL(Base):
    """Bayesian RVFL model class derived from class Base
    
       Parameters
       ----------
       obj: object
           any object containing a method fit (obj.fit()) and a method predict (obj.predict())
       n_hidden_features: int
           number of nodes in the hidden layer
       activation_name: str
           activation function: 'relu', 'tanh' or 'sigmoid'
       nodes_sim: str
           type of simulation for the nodes: 'sobol', 'hammersley', 'halton', 'uniform'
       bias: boolean
           indicates if the hidden layer contains a bias term (True) or not (False)
       direct_link: boolean
           indicates if the original predictors are included (True) in model's fitting or not (False)
       n_clusters: int
           number of clusters for 'kmeans' or 'gmm' clustering (could be 0: no clustering)
       type_clust: str
           type of clustering method: currently k-means ('kmeans') or Gaussian Mixture Model ('gmm')
       seed: int 
           reproducibility seed for nodes_sim=='uniform'
       s: float
           std. dev. of regression parameters in Bayesian Ridge Regression
       sigma: float
           std. dev. of residuals in Bayesian Ridge Regression
       beta: array-like
           fitted parameters of the Regression 
       Sigma: array-like
           covariance of the distribution of fitted parameters
    """
    
    # construct the object -----
    
    def __init__(self,
                 n_hidden_features=5, 
                 activation_name='relu',
                 nodes_sim='sobol',
                 bias=True,
                 direct_link=True, 
                 n_clusters=2,
                 type_clust='kmeans',
                 seed=123, 
                 s=0.1, sigma=0.05, 
                 beta=None, Sigma=None,
                 return_std = True):
                
        super().__init__(n_hidden_features, activation_name,
                         nodes_sim, bias, direct_link,
                         n_clusters, type_clust, seed)
        self.s = s 
        self.sigma = sigma
        self.beta = beta
        self.Sigma = Sigma
        self.return_std = return_std
    
    
    def get_params(self):
        
        return mx.merge_two_dicts(super().get_params(), 
                                  {"s": self.s, 
                                   "sigma": self.sigma,
                                   "return_std": self.return_std})

    
    def set_params(self, n_hidden_features=5, 
                   activation_name='relu', 
                   nodes_sim='sobol',
                   bias=True,
                   direct_link=True,
                   n_clusters=None,
                   type_clust='kmeans',
                   seed=123, 
                   s=0.1, sigma=0.05,
                   return_std = True):
        
        super().set_params(n_hidden_features = n_hidden_features, 
                           activation_name = activation_name, 
                           nodes_sim = nodes_sim,
                           bias = bias, direct_link = direct_link, 
                           n_clusters = n_clusters, type_clust = type_clust, 
                           seed = seed, return_std = return_std)
        self.s = s
        self.sigma = sigma
        self.return_std = return_std
        
    
    def fit(self, X, y, **kwargs):
        
        centered_y, scaled_Z = self.cook_training_set(y = y, X = X, **kwargs)
        
        fit_obj = lmf.beta_Sigma_hat_rvfl(X = scaled_Z, 
                                          y = centered_y, 
                                          s = self.s, 
                                          sigma = self.sigma,
                                          fit_intercept = False,
                                          return_cov = self.return_std)
        
        self.beta = fit_obj['beta_hat']
        
        if self.return_std == True: 
            self.Sigma = fit_obj['Sigma_hat']
        
        return self

    
    def predict(self, X, **kwargs):
        
        if len(X.shape) == 1: # one observation in the test set only
            n_features = X.shape[0]
            new_X = mo.rbind(X.reshape(1, n_features), 
                             np.ones(n_features).reshape(1, n_features))        
        
        if self.return_std == False:
            
            if len(X.shape) == 1:
            
                return (self.y_mean + np.dot(self.cook_test_set(new_X, 
                                                                **kwargs), 
                                         self.beta))[0]
            else:
                
                return (self.y_mean + np.dot(self.cook_test_set(X, **kwargs), 
                                         self.beta))
            
        else: # confidence interval required for preds?
            
            if len(X.shape) == 1:
            
                Z = self.cook_test_set(new_X, **kwargs)
                
                pred_obj = lmf.beta_Sigma_hat_rvfl(s = self.s, 
                                                   sigma = self.sigma, 
                                                   X_star = Z, 
                                                   return_cov = True, 
                                                   beta_hat_ = self.beta, 
                                                   Sigma_hat_ = self.Sigma) 
                
                return (self.y_mean + pred_obj['preds'][0], 
                        pred_obj['preds_std'][0]) 

            else:
                
                Z = self.cook_test_set(X, **kwargs)
                
                pred_obj = lmf.beta_Sigma_hat_rvfl(s = self.s, 
                                                   sigma = self.sigma, 
                                                   X_star = Z, 
                                                   return_cov = True,
                                                   beta_hat_ = self.beta, 
                                                   Sigma_hat_ = self.Sigma) 
                
                return (self.y_mean + pred_obj['preds'], 
                        pred_obj['preds_std']) 
   