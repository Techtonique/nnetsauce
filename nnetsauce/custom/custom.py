"""Custom model"""

# Authors: Thierry Moudiki 
#
# License: BSD 3

import numpy as np
import sklearn.model_selection as skm
import sklearn.metrics as skm2
from ..base import Base
from ..utils import matrixops as mo
from ..utils import misc as mx


class Custom(Base):
    """Custom model class derived from class Base
    
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
       type_fit: str
           'regression' or 'classification'
    """
    
    # construct the object -----
    
    def __init__(self, obj,
                 n_hidden_features=5, 
                 activation_name='relu',
                 a=0.01,
                 nodes_sim='sobol',
                 bias=True,
                 direct_link=True, 
                 n_clusters=2,
                 type_clust='kmeans',
                 seed=123, 
                 type_fit=None): 
                
        super().__init__(n_hidden_features = n_hidden_features, 
                         activation_name = activation_name, a = a,
                         nodes_sim = nodes_sim, bias = bias, 
                         direct_link = direct_link,
                         n_clusters = n_clusters, 
                         type_clust = type_clust, 
                         seed = seed)
        
        self.obj = obj
        self.type_fit = type_fit


    def get_params(self):
        
        return super().get_params()

    
    def set_params(self, n_hidden_features=5, 
                   activation_name='relu', 
                   a=0.01,
                   nodes_sim='sobol',
                   bias=True,
                   direct_link=True,
                   n_clusters=None,
                   type_clust='kmeans',
                   seed=123):
        
        super().set_params(n_hidden_features = n_hidden_features, 
                           activation_name = activation_name, a = a,
                           nodes_sim = nodes_sim, bias = bias, 
                           direct_link = direct_link, n_clusters = n_clusters, 
                           type_clust = type_clust, seed = seed)
 
    
    def fit(self, X, y, **kwargs):
        """Fit custom model to training data (X, y).
        
        Parameters
        ----------
        X: {array-like}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number 
            of samples and n_features is the number of features.
        
        y: array-like, shape = [n_samples]
               Target values.
    
        **kwargs: additional parameters to be passed to 
                  self.cook_training_set
               
        Returns
        -------
        self: object
        """
        if mx.is_factor(y) == False: 
            
            if self.type_fit is None: 
                self.type_fit = "regression"
                
            centered_y, scaled_Z = self.cook_training_set(y = y, X = X, 
                                                          **kwargs)
            
            self.obj.fit(scaled_Z, centered_y, **kwargs)
            
        else:
            
            if self.type_fit is None: 
                self.type_fit = "classification"
            
            scaled_Z = self.cook_training_set(y = y, X = X, 
                                              **kwargs)
            
            self.obj.fit(scaled_Z, y, **kwargs)
        
        return self

    
    def predict(self, X, **kwargs):
        """Predict test data X.
        
        Parameters
        ----------
        X: {array-like}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number 
            of samples and n_features is the number of features.
        
        **kwargs: additional parameters to be passed to 
                  self.cook_test_set
               
        Returns
        -------
        model predictions: {array-like}
        """
        
        if len(X.shape) == 1:
            
            n_features = X.shape[0]
            new_X = mo.rbind(X.reshape(1, n_features), 
                             np.ones(n_features).reshape(1, n_features))        
            
            if self.type_fit == "regression":
                
                return (self.y_mean + self.obj.predict(self.cook_test_set(new_X, 
                                                                             **kwargs), 
                                                   **kwargs))[0]
            else: # classification
                
                return (self.obj.predict(self.cook_test_set(new_X, **kwargs), 
                                         **kwargs))[0]
                
        else:
            
            if self.type_fit == "regression":
            
                return self.y_mean + self.obj.predict(self.cook_test_set(X, 
                                                                        **kwargs), 
                                               **kwargs)
            else:  # classification
                
                return self.obj.predict(self.cook_test_set(X, **kwargs), 
                                               **kwargs)
         

    def score(self, X, y, 
              scoring=None, **kwargs):
        """ Score the model on test set covariates X and response y. """
        
        preds = self.predict(X)
        
        if type(preds) == tuple: # if there are std. devs in the predictions
            preds = preds[0]    
        
        if mx.is_factor(y): # classification
            
            if scoring is None:
                scoring = 'accuracy'
            
            # check inputs 
            assert scoring in ('accuracy', 'average_precision', 
                               'brier_score_loss', 'f1', 'f1_micro',
                               'f1_macro', 'f1_weighted',  'f1_samples',
                               'neg_log_loss', 'precision', 'recall',
                               'roc_auc'), \
                               "'scoring' should be in ('accuracy', 'average_precision', \
                               'brier_score_loss', 'f1', 'f1_micro', \
                               'f1_macro', 'f1_weighted',  'f1_samples', \
                               'neg_log_loss', 'precision', 'recall', \
                               'roc_auc')"
            
            scoring_options = {
                'accuracy': skm2.accuracy_score,
                'average_precision': skm2.average_precision_score,
                'brier_score_loss': skm2.brier_score_loss,
                'f1': skm2.f1_score,
                'f1_micro': skm2.f1_score,
                'f1_macro': skm2.f1_score,
                'f1_weighted': skm2.f1_score,
                'f1_samples': skm2.f1_score,
                'neg_log_loss': skm2.log_loss,
                'precision': skm2.precision_score,
                'recall': skm2.recall_score,
                'roc_auc': skm2.roc_auc_score
            } 
            
        else: # regression
            
            if scoring is None:
                scoring = 'neg_mean_squared_error'
            
            # check inputs 
            assert scoring in ('explained_variance', 'neg_mean_absolute_error', 
                               'neg_mean_squared_error', 'neg_mean_squared_log_error', 
                               'neg_median_absolute_error', 'r2'), \
                               "'scoring' should be in ('explained_variance', 'neg_mean_absolute_error', \
                               'neg_mean_squared_error', 'neg_mean_squared_log_error', \
                               'neg_median_absolute_error', 'r2')"
            
            scoring_options = {
                'explained_variance': skm2.explained_variance_score,
                'neg_mean_absolute_error': skm2.median_absolute_error,
                'neg_mean_squared_error': skm2.mean_squared_error,
                'neg_mean_squared_log_error': skm2.mean_squared_log_error, 
                'neg_median_absolute_error': skm2.median_absolute_error,
                'r2': skm2.r2_score
                } 
        
        return scoring_options[scoring](y, preds, **kwargs)    
    
    
    def cross_val_score(self, X, y, groups=None, scoring=None, cv=None,
                        n_jobs=1, verbose=0, fit_params=None,
                        pre_dispatch='2*n_jobs',  **kwargs):
        
        """Cross-validation scores (sklearn style).
        
        Parameters
        ----------
        Same as sklearn.model_selection.cross_val_score
          
        Returns
        -------
        scores : array of float, shape=(len(list(cv)),)
            Array of scores of the estimator for each run of the cross validation.
        """

        if mx.is_factor(y) == False: # regression
            
            if self.type_fit is None: 
                self.type_fit = "regression"
                
            centered_y, scaled_Z = self.cook_training_set(y = y, X = X, **kwargs)
            return skm.cross_val_score(self.obj, X = scaled_Z, y = centered_y, 
                                       groups = groups, scoring = scoring, cv = cv, 
                                       n_jobs = n_jobs, verbose = verbose, 
                                       fit_params = fit_params, 
                                       pre_dispatch = pre_dispatch)
            
        else: # classification
            
            if self.type_fit is None: 
                self.type_fit = "classification"
                
            scaled_Z = self.cook_training_set(y = y, X = X, **kwargs)
            return skm.cross_val_score(self.obj, X = scaled_Z, y = y, groups = groups, 
                                  scoring = scoring, cv = cv, n_jobs = n_jobs, 
                                  verbose = verbose, fit_params = fit_params, 
                                  pre_dispatch = pre_dispatch)
        