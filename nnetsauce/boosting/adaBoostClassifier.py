"""AdaBoosting model"""

# Authors: Thierry Moudiki
#
# License: BSD 3

import numpy as np
import copy as cp
import sklearn.metrics as skm2
from .bst import Boosting
from ..custom import CustomClassifier
from ..utils import matrixops as mo
from ..utils import misc as mx
from ..utils import Progbar
from sklearn.base import ClassifierMixin
from scipy.linalg import norm
from scipy.special import xlogy

class AdaBoostClassifier(Boosting, ClassifierMixin):
    """AdaBoost Classification (SAMME) model class derived from class Boosting
    
       Parameters
       ----------
       obj: object
           any object containing a method fit (obj.fit()) and a method predict 
           (obj.predict())
       n_estimators: int
           number of boosting iterations
       learning_rate: float
           
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
        n_estimators=10,
        learning_rate=0.1,
        n_hidden_features=1,
        activation_name="relu",
        a=0.01,
        nodes_sim="sobol",
        bias=True,
        dropout=0,
        direct_link=False,
        n_clusters=2,
        type_clust="kmeans",
        type_scaling=("std", "std", "std"),
        col_sample=1,
        row_sample=1,
        seed=123,
        verbose=1
    ):

        super().__init__(
            obj=obj,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
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
        self.alpha = []
        self.w = []
        self.base_learners = []
        self.verbose = verbose


    def fit(self, X, y, method="SAMME", **kwargs):
        """Fit Boosting model to training data (X, y).
        
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
        
        assert mx.is_factor(
            y
        ), "y must contain only integers"     

        assert method in ("SAMME", "SAMME.R"),\
        "method must be either SAMME or SAMME.R"                                                                                           
        
        
        base_learner = CustomClassifier(self.obj,
        n_hidden_features=self.n_hidden_features,
        activation_name=self.activation_name,
        a=self.a,
        nodes_sim=self.nodes_sim,
        bias=self.bias,
        dropout=self.dropout,
        direct_link=self.direct_link,
        n_clusters=self.n_clusters,
        type_clust=self.type_clust,
        type_scaling=self.type_scaling,
        col_sample=self.col_sample,
        row_sample=self.row_sample,
        seed=self.seed)
        
        
        # training 
        n, p = X.shape   
        K = len(np.unique(y))
        self.n_classes = K
        #ws = []
        w_m = np.repeat(1/n, n) # (N, 1)
        #ws.append(cp.deepcopy(w_m.tolist()))
        weighted_X = w_m[:, None]*X # (N, K)    
        self.method = method
        
        
        if self.verbose == 1:
            pbar = Progbar(self.n_estimators)                  
        
        
        if self.method == "SAMME":        
            
            err_m = 1e6            
            #self.alpha.append(self.learning_rate*1.0)
            self.alpha.append(1.0)        
            
            for m in range(self.n_estimators): 
                                                        
                preds = base_learner.fit(weighted_X, y, **kwargs).predict(weighted_X)
                self.base_learners.append(cp.deepcopy(base_learner))
                cond = (y != preds)
                            
                err_m = np.mean(w_m*cond)
                alpha_m = self.learning_rate*(np.log((1 - err_m)/err_m) + np.log(K - 1))            
                self.alpha.append(alpha_m)
                
                w_m *= np.exp(alpha_m*cond) 
                w_m /= np.sum(w_m)
                #ws.append(cp.deepcopy(w_m.tolist()))            
    
                weighted_X = w_m[:, None]*X
                base_learner.set_params(seed = self.seed + (m + 1)*1000)      
                
                if self.verbose == 1:              
                    pbar.update(m)
            
            if self.verbose == 1:
                pbar.update(self.n_estimators)                    
    
            return self


        if self.method == "SAMME.R": 
            
            Y = mo.one_hot_encode2(y, self.n_classes)
            
            for m in range(self.n_estimators):
                
                probs = base_learner.fit(weighted_X, y, **kwargs).predict_proba(weighted_X) 
                
                np.clip(a = probs, a_min = np.finfo(probs.dtype).eps, 
                        a_max = 1.0, out=probs)                                                  
                
                self.base_learners.append(cp.deepcopy(base_learner))
                
                w_m *= np.exp(-1.0*self.learning_rate*(1.0 - 1.0/self.n_classes)\
                *xlogy(Y, probs).sum(axis=1))
                
                w_m /= np.sum(w_m)
    
                weighted_X = w_m[:, None]*X
                
                base_learner.set_params(seed = self.seed + (m + 1)*1000)      
                
                if self.verbose == 1:              
                    pbar.update(m)
                
            if self.verbose == 1:
                pbar.update(self.n_estimators) 
                
        
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
        
        return self.predict_proba(X, **kwargs).argmax(axis=1)


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
        
        if self.method == "SAMME.R":
            
            ensemble_learner = 0    
        
            if self.verbose == 1:
                pbar = Progbar(self.n_estimators)            
            
            for idx, base_learner in enumerate(self.base_learners):             
                                                                 
                probs = base_learner.predict_proba(X, **kwargs)
                
                np.clip(a = probs, a_min = np.finfo(probs.dtype).eps, 
                    a_max = 1.0, out = probs)    

                log_preds_proba = np.log(probs)        
                
                ensemble_learner += (self.n_classes - 1)*(log_preds_proba -\
                                    log_preds_proba.mean(axis=1)[:, None])
                
                if self.verbose == 1:            
                    pbar.update(idx)
    
            if self.verbose == 1:        
                pbar.update(self.n_estimators)
            
            sum_ensemble = ensemble_learner.sum(axis=1)
            
            np.clip(a = sum_ensemble, a_min = np.finfo(sum_ensemble.dtype).eps, 
                    a_max = None, out = sum_ensemble)    
            
            return ensemble_learner/sum_ensemble[:, None]
            
        # self.method == "SAMME"
        ensemble_learner = 0    
        
        if self.verbose == 1:
            pbar = Progbar(self.n_estimators)            
        
        for idx, base_learner in enumerate(self.base_learners): 
        
            preds = base_learner.predict(X, **kwargs)    
            
            ensemble_learner += self.alpha[idx]*mo.one_hot_encode2(preds,\
                                          self.n_classes)
            
            if self.verbose == 1:            
                pbar.update(idx)

        if self.verbose == 1:        
            pbar.update(self.n_estimators)
        
        sum_ensemble = ensemble_learner.sum(axis=1)
            
        np.clip(a = sum_ensemble, a_min = np.finfo(sum_ensemble.dtype).eps, 
                    a_max = None, out = sum_ensemble)    
            
        return ensemble_learner/sum_ensemble[:, None]


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
