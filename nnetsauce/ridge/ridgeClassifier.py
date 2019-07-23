"""Ridge model for classification"""

# Authors: Thierry Moudiki
#
# License: BSD 3

import autograd.numpy as np
from autograd import grad
from scipy.optimize import minimize
import sklearn.metrics as skm2
from .ridge import Ridge
from ..utils import matrixops as mo
from ..utils import misc as mx
#from scipy.special import logsumexp
from sklearn.base import ClassifierMixin
from time import time


class RidgeClassifier(Ridge, ClassifierMixin):
    """Ridge Classification model class derived from class Ridge
    
       Parameters
       ----------
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
        lambda1=0.1,
        lambda2=0.1,
        seed=123,
    ):

        super().__init__(
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
            lambda1=lambda1,
            lambda2=lambda2,
            seed=seed,
        )

        self.type_fit = "classification"

    
    def loglik(self, X, Y, beta, grad_hess=True, **kwargs):
        """Log-likelihood for training data (X, y).
        
        Parameters
        ----------
        X: {array-like}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number 
            of samples and n_features is the number of features.
        
        y: array-like, shape = [n_samples]
               Target values.
    
        beta: regression coeffs (beta11, ..., beta1p, ..., betaK1, ..., betaKp)
              for K classes and p covariates.
    
        **kwargs: additional parameters to be passed to 
                  self.cook_training_set or self.obj.fit
               
        Returns
        -------
        """                
                       
        # total number of covariates
        p = X.shape[1]
        
        # initial number of covariates
        init_p = p - self.n_hidden_features
        
        # (p, K)
        B = beta.reshape(Y.shape[1], p).T     
        
        # (n, K)
        XB = np.dot(X, B)
        
        def logsumexp(x):
            """Numerically stable log(sum(exp(x))), also defined in scipy.special"""
            max_x = np.max(x)
            return max_x + np.log(np.sum(np.exp(x - max_x), axis = 1))
            
        
        return -np.mean((np.sum(Y*XB, axis=1) - logsumexp(XB))) +\
        0.5*self.lambda1*mo.squared_norm(B[0:init_p,:]) +\
        0.5*self.lambda2*mo.squared_norm(B[init_p:p,:])
 
    
    # to be merged with loglik
    # to be merged with loglik
    # to be merged with loglik
    def probas(self, X, Y, beta, **kwargs):
        # total number of covariates
        p = X.shape[1]
        
        # (p, K)
        B = beta.reshape(Y.shape[1], p).T 
        
        # (n, K)
        XB = np.dot(X, B)
        
        # (n, K)
        exp_XB = np.exp(XB)                
        
        # (n, K)
        return exp_XB/exp_XB.sum(axis=1)[:, None]
    
    # to be merged with loglik
    # to be merged with loglik
    # to be merged with loglik
    def grad_hess(self, X, Y, beta, **kwargs):
        
        # nobs
        n, K = Y.shape
        
        # total number of covariates
        p = X.shape[1]
        
        # initial number of covariates
        init_p = p - self.n_hidden_features
        
        # (N, K)
        probs = self.probas(X, Y, beta, **kwargs)
        
        # (p, K)
        B = beta.reshape(K, p).T
                
        # (Y - p) -> (n, K)
        # X -> (n, p)        
        # (K, n) %*% (n, p) -> (K, p)
        # make flat?
        res = - np.dot((Y - probs).T, X)/n +\
    self.lambda1*B[0:init_p,:].sum(axis=0)[:, None] +\
    self.lambda2*B[init_p:p,:].sum(axis=0)[:, None]
                
        return res.flatten() 
        
    # trust-ncg
    # newton-cg
    # L-BFGS-B
    def fit(self, X, y, solver = "L-BFGS-B", **kwargs):
        """Fit Ridge model to training data (X, y).
        
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
        
        assert mx.is_factor(y), "y must contain only integers"        
        
        output_y, scaled_Z = self.cook_training_set(
            y=y, X=X, **kwargs
        )
        
        self.n_classes = len(np.unique(y))
        
        Y = mo.one_hot_encode2(output_y)
        
        # optimize for beta, minimize self.loglik (maximize loglik) -----        
        def loglik_objective(x):
            return(self.loglik(X = scaled_Z, 
                               Y = Y, # one-hot encoded response
                               beta = x))
        def grad_objective(x):
            return(self.grad_hess(X= scaled_Z, 
                                  Y = Y, 
                                  beta = x))
            
        x0 = np.zeros(scaled_Z.shape[1]*self.n_classes)
        
        grad_loglik = grad(loglik_objective)
        
        hess_loglik = grad(grad_objective)
        
        print("\n")
        print("loglik_objective(x0)")        
        print(loglik_objective(x0))
        print("\n")
        print("grad_objective(x0)")            
        start = time()
        print(grad_objective(x0))            
        end = time()
        print(f"Elapsed {end - start}")
        print("\n")        
        print("grad_loglik(x0)")            
        start = time()
        print(grad_loglik(x0))  
        end = time()          
        print(f"Elapsed {end - start}")
        print("\n")
        
        
        self.beta = minimize(fun = loglik_objective, 
                             x0 = np.zeros(scaled_Z.shape[1]*self.n_classes), 
                             jac = grad_objective,    
                             #hess = hess_loglik,
                             method=solver).x  
                
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

        return np.argmax(self.predict_proba(X, **kwargs), 
                         axis = 1)


    def predict_proba(self, X, **kwargs):
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

            n_features = X.shape[0]
            new_X = mo.rbind(
                X.reshape(1, n_features),
                np.ones(n_features).reshape(
                    1, n_features
                ),
            )
            
            Z = self.cook_test_set(new_X, **kwargs)                        

        else:

            Z = self.cook_test_set(X, **kwargs)
        
        ZB = mo.safe_sparse_dot(Z, self.beta.reshape(self.n_classes, 
                                         X.shape[1] + self.n_hidden_features).T)
        
        exp_ZB = np.exp(ZB)
        
        return exp_ZB/exp_ZB.sum(axis=1)[:, None]
                      
        

    def score(self, X, y, scoring=None, **kwargs):
        """ Score the model on test set covariates X and response y. """

        preds = self.predict(X)

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

