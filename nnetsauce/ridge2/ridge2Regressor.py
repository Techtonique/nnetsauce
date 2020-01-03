"""Ridge model for regression with 2 regularization parameters"""

# Authors: Thierry Moudiki
#
# License: BSD 3

import numpy as np
from scipy.optimize import minimize
import sklearn.metrics as skm
from .ridge2 import Ridge2
from ..utils import matrixops as mo
from ..utils import misc as mx
from sklearn.base import RegressorMixin
from scipy.special import logsumexp
from scipy.linalg import pinv


class Ridge2Regressor(Ridge2, RegressorMixin):
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
       col_sample: float
           percentage of covariates randomly chosen for training   
       row_sample: float
           percentage of rows chosen for training, by stratified bootstrapping    
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
            n_clusters=n_clusters,
            cluster_encode=cluster_encode,
            type_clust=type_clust,
            type_scaling=type_scaling,
            col_sample=col_sample,
            row_sample=row_sample,
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

        centered_y, scaled_Z = self.cook_training_set(y=y, X=X, **kwargs)

        n_X, p_X = X.shape
        n_Z, p_Z = scaled_Z.shape

        if self.n_clusters > 0:
            if self.encode_clusters == True:
                n_features = p_X + self.n_clusters
            else:
                n_features = p_X + 1
        else:
            n_features = p_X

        X_ = scaled_Z[:, 0:n_features]
        Phi_X_ = scaled_Z[:, n_features:p_Z]

        B = mo.crossprod(X_) + self.lambda1 * np.diag(np.repeat(1, n_features))
        C = mo.crossprod(Phi_X_, X_)
        D = mo.crossprod(Phi_X_) + self.lambda2 * np.diag(
            np.repeat(1, Phi_X_.shape[1])
        )
        B_inv = pinv(B)
        W = np.dot(C, B_inv)
        S_mat = D - mo.tcrossprod(W, C)
        S_inv = pinv(S_mat)
        Y = np.dot(S_inv, W)
        inv = mo.rbind(
            mo.cbind(B_inv + mo.crossprod(W, Y), -np.transpose(Y)),
            mo.cbind(-Y, S_inv),
        )

        self.beta = np.dot(inv, mo.crossprod(scaled_Z, centered_y))

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
            new_X = mo.rbind(
                X.reshape(1, n_features),
                np.ones(n_features).reshape(1, n_features),
            )

            return (
                self.y_mean
                + np.dot(self.cook_test_set(new_X, **kwargs), self.beta)
            )[0]

        return self.y_mean + np.dot(self.cook_test_set(X, **kwargs), self.beta)

    def score(self, X, y, scoring=None, **kwargs):
        """ Score the model on test set covariates X and response y. """

        preds = self.predict(X)

        if type(preds) == tuple:  # if there are std. devs in the predictions
            preds = preds[0]

        if scoring is None:
            scoring = "neg_mean_squared_error"

        # check inputs
        assert scoring in (
            "explained_variance",
            "neg_mean_absolute_error",
            "neg_mean_squared_error",
            "neg_mean_squared_log_error",
            "neg_median_absolute_error",
            "r2",
        ), "'scoring' should be in ('explained_variance', 'neg_mean_absolute_error', \
                           'neg_mean_squared_error', 'neg_mean_squared_log_error', \
                           'neg_median_absolute_error', 'r2')"

        scoring_options = {
            "explained_variance": skm.explained_variance_score,
            "neg_mean_absolute_error": skm.median_absolute_error,
            "neg_mean_squared_error": skm.mean_squared_error,
            "neg_mean_squared_log_error": skm.mean_squared_log_error,
            "neg_median_absolute_error": skm.median_absolute_error,
            "r2": skm.r2_score,
        }

        return scoring_options[scoring](y, preds, **kwargs)
