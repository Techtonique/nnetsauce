"""RNN CLassifier"""

# Authors: Thierry Moudiki
#
# License: BSD 3

import numpy as np
import sklearn.metrics as skm2
from .rnn import RNN
#from ..utils import matrixops as mo
from sklearn.base import ClassifierMixin
from ..utils import Progbar

class RNNClassifier(RNN, ClassifierMixin):
    """RNN Classification model class derived from class RNN
    
       Parameters
       ----------
       obj: object
           any object containing a method fit (obj.fit()) and a method predict 
           (obj.predict())
       alpha: float
           smoothing parameter
       window: int
           size of training window
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
       dropout: float
           regularization parameter; (random) percentage of nodes dropped out 
           of the training
       direct_link: boolean
           indicates if the original predictors are included (True) in model's 
           fitting or not (False)
       n_clusters: int
           number of clusters for 'kmeans' or 'gmm' clustering (could be 0: 
               no clustering)
       type_clust: str
           type of clustering method: currently k-means ('kmeans') or Gaussian 
           Mixture Model ('gmm')
       type_scaling: a tuple of 3 strings
           scaling methods for inputs, hidden layer, and clustering respectively
           (and when relevant). 
           Currently available: standardization ('std') or MinMax scaling ('minmax')
       col_sample: float
           percentage of covariates randomly chosen for training   
       row_sample: float
           percentage of rows chosen for training, by stratified bootstrapping    
       seed: int 
           reproducibility seed for nodes_sim=='uniform'
    """

    # construct the object -----

    def __init__(
        self,
        obj,
        alpha=0.5,
        window=2,
        n_hidden_features=5,
        activation_name="relu",
        a=0.01,
        nodes_sim="sobol",
        bias=True,
        dropout=0,
        direct_link=True,
        n_clusters=2,
        type_clust="kmeans",
        type_scaling=("std", "std", "std"),
        col_sample=1,
        row_sample=1,
        seed=123,
    ):

        super().__init__(
            obj=obj,
            alpha=alpha,
            n_hidden_features=n_hidden_features,
            activation_name=activation_name,
            a=a,
            nodes_sim=nodes_sim,
            bias=bias,
            dropout=dropout,
            direct_link=direct_link,
            n_clusters=n_clusters,
            type_clust=type_clust,
            type_scaling=type_scaling,
            col_sample=col_sample,
            row_sample=row_sample,
            seed=seed,
        )

        self.type_fit = "classification"  
        self.window = window


    def fit(self, inputs, targets=None, 
            scoring=None, n_params=None, verbose=0): 
        """ Fit RNN to inputs and targets. """

        steps = inputs.shape[0]
        
        assert (steps > 0), "inputs.shape[0] must be > 0"                 
        
        self.steps = steps
        
        
        if targets is not None:
            assert (steps == targets.shape[0]), \
            "'inputs' and 'targets' must contain the same number of steps"        
            self.last_target = np.transpose(targets[(steps-self.window):steps, :])
        else:
            self.last_target = np.transpose(inputs[(steps-self.window):steps, :])
        
        
        # loss obtained by fitting on training set
        loss = 0                        
        
        if targets is not None: 
            
            j = self.window - 1   
            n_steps = steps - self.window + 1 
            
            if verbose == 1:
                pbar = Progbar(target=n_steps)                               
            
            for i in range(n_steps):   
                
                print("i= \n")                       
                print(i)  
                batch_index = range(i, i + self.window)
                self.fit_step(X = inputs[batch_index,:], y = targets[j,:])                               
                loss += self.score_step(X = inputs[batch_index,:], 
                                        y = targets[j,:], 
                                        scoring = scoring)
                
                if verbose == 1:
                    pbar.update(i)
                    
                j += 1                            

            if verbose == 1:
                pbar.update(n_steps)
            
        else: # targets is None
            
            j = self.window
            n_steps = steps - self.window
            
            if verbose == 1:
                pbar = Progbar(target=n_steps)        
                
            for i in range(n_steps):
                print("i= \n")                       
                print(i)  
                batch_index = range(i, i + self.window)
                self.fit_step(X = inputs[batch_index,:], y = inputs[j,:])                    
                loss += self.score_step(X = inputs[batch_index,:], 
                                        y = inputs[j,:], 
                                        scoring = scoring)
                
                if verbose == 1:
                    pbar.update(i)
                
                j += 1   
            
            if verbose == 1:
                pbar.update(n_steps)
                
        return loss/n_steps
    
    
    def predict(self, h=5, **kwargs):
                   
        assert (self.steps > 0), "method 'fit' must be called first"
        
        n_series = self.last_target.shape[0]
        
        n_res = h + self.window
        
        res = np.zeros((n_series, n_res))
        
        print("self.last_target")
        print(self.last_target)            
        
        print("self.last_target.shape")
        print(self.last_target.shape)    
        
        try:                        
            res[:, 0:self.window] = self.last_target.reshape(self.last_target.shape[0], self.last_target.shape[1])
        except:
            res[:, 0:self.window] = self.last_target.reshape(-1, 1)
                    
        if 'return_std' not in kwargs:             
            
            for i in range(self.window, self.window+h):            
            
                res[:,i] = self.predict_step(X = res[:,(i-self.window):i], 
                                    **kwargs)
                
            return np.transpose(res)[(n_res-h):n_res,:]


     # add predict_proba
     # add predict_proba
     # add predict_proba
     # add predict_proba
     # add predict_proba        


    def fit_step(self, X, y, **kwargs):
        """Fit RNN model to training data (X, y).
        
        Parameters
        ----------
        X: {array-like}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number 
            of samples and n_features is the number of features.
        
        y: array-like, shape = [n_samples]
               Target values.
    
        **kwargs: additional parameters to be passed to 
                  self.cook_training_set or self.obj.fit
               
        Returns
        -------
        self: object
        """

        if (len(X.shape) == 1):
            X = X.reshape(-1, 1)      
        else:
            X = np.transpose(X)
        
        print("========== \n")
        print("X: \n")    
        print(X)
        print("\n")
        print("X.shape: \n")    
        print(X.shape)
        print("\n")
        print("y: \n")    
        print(y)
        print("\n")
        
        # calls 'create_layer' from parent RNN: obtains centered_y, updates state H.
        # 'scaled_Z' is not used, but H
        output_y, scaled_Z = self.cook_training_set(
            y=y, X=X, **kwargs
        )
        
        self.obj.fit(X = scaled_Z, y = output_y, **kwargs)

        return self
       

    def predict_step(self, X, **kwargs):
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
            X = X.reshape(-1, 1)      
        
        return self.obj.predict(
                self.cook_test_set(X, **kwargs), **kwargs
            )

        
    def predict_proba_step(self, X, **kwargs):
        """Predict probabilities for test data X.
        
        Parameters
        ----------
        X: {array-like}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number 
            of samples and n_features is the number of features.
        
        **kwargs: additional parameters to be passed to 
                  self.cook_test_set
               
        Returns
        -------
        probability estimates for test data: {array-like}        
        """
        
        if len(X.shape) == 1:            
            X = X.reshape(-1, 1)      
        
        return self.obj.predict_proba(
                    self.cook_test_set(X, **kwargs),
                    **kwargs
                )                


    def score_step(self, X, y, scoring=None, **kwargs):
        """ Score the model on test set covariates X and response y. """

        preds = self.predict_step(X)

        if scoring is None:
            scoring = "accuracy"

        # check inputs
        assert scoring in (
            "accuracy",
            "average_precision",
            "brier_score_loss",
            "f1",
            "f1_micro",
            "f1_macro",
            "f1_weighted",
            "f1_samples",
            "neg_log_loss",
            "precision",
            "recall",
            "roc_auc",
        ), "'scoring' should be in ('accuracy', 'average_precision', \
                           'brier_score_loss', 'f1', 'f1_micro', \
                           'f1_macro', 'f1_weighted',  'f1_samples', \
                           'neg_log_loss', 'precision', 'recall', \
                           'roc_auc')"

        scoring_options = {
            "accuracy": skm2.accuracy_score,
            "average_precision": skm2.average_precision_score,
            "brier_score_loss": skm2.brier_score_loss,
            "f1": skm2.f1_score,
            "f1_micro": skm2.f1_score,
            "f1_macro": skm2.f1_score,
            "f1_weighted": skm2.f1_score,
            "f1_samples": skm2.f1_score,
            "neg_log_loss": skm2.log_loss,
            "precision": skm2.precision_score,
            "recall": skm2.recall_score,
            "roc_auc": skm2.roc_auc_score,
        }

        return scoring_options[scoring](y, preds, **kwargs)