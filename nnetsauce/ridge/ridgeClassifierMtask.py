"""Ridge model for regression"""

# Authors: Thierry Moudiki
#
# License: BSD 3

import numpy as np
from scipy.optimize import minimize
import sklearn.metrics as skm2
from .ridge import Ridge
from ..utils import matrixops as mo
from ..utils import misc as mx
from sklearn.base import ClassifierMixin
from scipy.special import logsumexp
from scipy.linalg import pinv


class RidgeClassifierMtask(Ridge, ClassifierMixin):
    """Ridge regression model class with 2 regularization parameters derived from class Ridge
    
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
       n_clusters: int
           number of clusters for 'kmeans' or 'gmm' clustering (could be 0: 
               no clustering)
       cluster_encode: bool
           defines how the variable containing clusters is treated (default is one-hot)
           if `False`, then labels are used, without one-hot encoding
       type_clust: str
           type of clustering method: currently k-means ('kmeans') or Gaussian 
           Mixture Model ('gmm')
       type_scaling: a tuple of 3 strings
           scaling methods for inputs, hidden layer, and clustering respectively
           (and when relevant). 
           Currently available: standardization ('std') or MinMax scaling ('minmax')
       lambda1: float
           regularization parameter on direct link
       lambda2: float
           regularization parameter on hidden layer
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
        n_clusters=2,
        cluster_encode=True,
        type_clust="kmeans",
        type_scaling=("std", "std", "std"),
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
            n_clusters=n_clusters,
            cluster_encode=cluster_encode,
            type_clust=type_clust,
            type_scaling=type_scaling,
            col_sample=1,
            row_sample=1,
            lambda1=lambda1,
            lambda2=lambda2,
            seed=seed,
        )

        self.type_fit = "regression"

    def fit(self, X, y, **kwargs):
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
        
        n_X, p_X = X.shape
        n_Z, p_Z = scaled_Z.shape
        
        self.n_classes = len(np.unique(y))
        
        # multitask response
        Y = mo.one_hot_encode2(y, self.n_classes)                
        
        if self.n_clusters > 0:
            if self.encode_clusters == True:
                n_features = p_X + self.n_clusters
            else:
                n_features = p_X + 1
        else:
            n_features = p_X

        X_ = scaled_Z[:, 0:n_features]
        Phi_X_ = scaled_Z[:, n_features:p_Z]                

        B = mo.crossprod(X_) + self.lambda1 * np.diag(
            np.repeat(1, X_.shape[1])
        )
        C = mo.crossprod(Phi_X_, X_)
        D = mo.crossprod(Phi_X_) + self.lambda2 * np.diag(
            np.repeat(1, Phi_X_.shape[1])
        )
        B_inv = pinv(B)
        W = np.dot(C, B_inv)
        S_mat = D - mo.tcrossprod(W, C)
        S_inv = pinv(S_mat)
        Y2 = np.dot(S_inv, W)
        inv = mo.rbind(
            mo.cbind(
                B_inv + mo.crossprod(W, Y2), -np.transpose(Y2)
            ),
            mo.cbind(-Y2, S_inv),
        )
            
        self.beta = np.dot(
            inv, mo.crossprod(scaled_Z, Y)
        )

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

        return np.argmax(
            self.predict_proba(X, **kwargs), axis=1
        )

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
                np.ones(n_features).reshape(1, n_features),
            )

            Z = self.cook_test_set(new_X, **kwargs)

        else:

            Z = self.cook_test_set(X, **kwargs)

        ZB = mo.safe_sparse_dot(Z, self.beta)
        
        exp_ZB = np.exp(ZB)

        return exp_ZB / exp_ZB.sum(axis=1)[:, None]

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
