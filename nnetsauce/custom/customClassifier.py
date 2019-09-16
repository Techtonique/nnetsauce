"""Custom model"""

# Authors: Thierry Moudiki
#
# License: BSD 3

import numpy as np
import sklearn.metrics as skm2
from .custom import Custom
from ..utils import matrixops as mo
from sklearn.base import ClassifierMixin


class CustomClassifier(Custom, ClassifierMixin):
    """Custom Classification model class derived from class Custom
    
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
       dropout: float
           regularization parameter; (random) percentage of nodes dropped out 
           of the training
       direct_link: boolean
           indicates if the original predictors are included (True) in model's 
           fitting or not (False)
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
       seed: int 
           reproducibility seed for nodes_sim=='uniform'
    """

    # construct the object -----

    def __init__(
        self,
        obj,
        n_hidden_features=5,
        activation_name="relu",
        a=0.01,
        nodes_sim="sobol",
        bias=True,
        dropout=0,
        direct_link=True,
        n_clusters=2,
        cluster_encode=True,
        type_clust="kmeans",
        type_scaling=("std", "std", "std"),
        col_sample=1,
        row_sample=1,
        seed=123,
    ):

        super().__init__(
            obj=obj,
            n_hidden_features=n_hidden_features,
            activation_name=activation_name,
            a=a,
            nodes_sim=nodes_sim,
            bias=bias,
            dropout=dropout,
            direct_link=direct_link,
            n_clusters=n_clusters,
            cluster_encode=cluster_encode,
            type_clust=type_clust,
            type_scaling=type_scaling,
            col_sample=col_sample,
            row_sample=row_sample,
            seed=seed,
        )

        self.type_fit = "classification"

    def fit(self, X, y, sample_weight=None, **kwargs):
        """Fit custom model to training data (X, y).
        
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

        output_y, scaled_Z = self.cook_training_set(
            y=y, X=X, **kwargs
        )

        # if sample_weights, else: (must use self.row_index)

        if sample_weight is not None:

            self.obj.fit(
                scaled_Z,
                output_y,
                sample_weight=sample_weight[self.index_row],
                **kwargs
            )

            return self

        # if sample_weight is None:
        self.obj.fit(scaled_Z, output_y, **kwargs)

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
                self.obj.predict(
                    self.cook_test_set(new_X, **kwargs),
                    **kwargs
                )
            )[0]

        return self.obj.predict(
            self.cook_test_set(X, **kwargs), **kwargs
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

            return (
                self.obj.predict_proba(
                    self.cook_test_set(new_X, **kwargs),
                    **kwargs
                )
            )[0]

        return self.obj.predict_proba(
            self.cook_test_set(X, **kwargs), **kwargs
        )

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
