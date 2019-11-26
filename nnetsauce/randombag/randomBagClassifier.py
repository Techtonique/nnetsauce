"""Random 'Forest' model"""

# Authors: Thierry Moudiki
#
# License: BSD 3

import numpy as np
import sklearn.metrics as skm2
from .bag import RandomBag
from ..custom import CustomClassifier
from ..utils import misc as mx
from ..utils import Progbar
from sklearn.base import ClassifierMixin
import pickle
from joblib import Parallel, delayed
from tqdm import tqdm


class RandomBagClassifier(RandomBag, ClassifierMixin):
    """Random 'Forest' Classification model class derived from class Boosting
    
       Parameters
       ----------
       obj: object
           any object containing a method fit (obj.fit()) and a method predict 
           (obj.predict())
       n_estimators: int
           number of boosting iterations
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
        n_estimators=10,
        n_hidden_features=1,
        activation_name="relu",
        a=0.01,
        nodes_sim="sobol",
        bias=True,
        dropout=0,
        direct_link=False,
        n_clusters=2,
        cluster_encode=True,
        type_clust="kmeans",
        type_scaling=("std", "std", "std"),
        col_sample=1,
        row_sample=1,
        n_jobs=None,
        seed=123,
        verbose=1,
    ):

        super().__init__(
            obj=obj,
            n_estimators=n_estimators,
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
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.voter = dict.fromkeys(range(n_estimators))

    def fit(self, X, y, **kwargs):
        """Fit Random 'Forest' model to training data (X, y).
        
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
            y.tolist()
        ), "y must contain only integers"

        # training
        n, p = X.shape
        self.n_classes = len(np.unique(y))

        base_learner = CustomClassifier(
            self.obj,
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
            seed=self.seed,
        )

        if self.verbose == 1:
            pbar = Progbar(self.n_estimators)

        # 1 - Sequential training -----

        if self.n_jobs is None:

            for idx, m in enumerate(
                range(self.n_estimators)
            ):

                try:

                    base_learner.fit(X, y, **kwargs)

                    self.voter.update(
                        {
                            m: pickle.loads(
                                pickle.dumps(
                                    base_learner, -1
                                )
                            )
                        }
                    )

                    base_learner.set_params(
                        seed=self.seed + (m + 1) * 1000
                    )

                    if self.verbose == 1:
                        pbar.update(m)

                except:

                    if self.verbose == 1:
                        pbar.update(m)

                    continue

            if self.verbose == 1:
                pbar.update(self.n_estimators)

            self.n_estimators = len(self.voter)

            return self

        # 2 - Parallel training -----

        # if self.n_jobs is not None:
        def fit_estimators(m):
            base_learner = CustomClassifier(
                self.obj,
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
                seed=self.seed + m * 1000,
            )
            base_learner.fit(X, y, **kwargs)
            self.voter.update(
                {
                    m: pickle.loads(
                        pickle.dumps(base_learner, -1)
                    )
                }
            )

        if self.verbose == 1:

            Parallel(n_jobs=self.n_jobs, prefer="threads")(
                delayed(fit_estimators)(m)
                for m in tqdm(range(self.n_estimators))
            )
        else:

            Parallel(n_jobs=self.n_jobs, prefer="threads")(
                delayed(fit_estimators)(m)
                for m in range(self.n_estimators)
            )

        self.n_estimators = len(self.voter)

        return self

    def predict(self, X, weights=None, **kwargs):
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
        return self.predict_proba(
            X, weights, **kwargs
        ).argmax(axis=1)

    def predict_proba(self, X, weights=None, **kwargs):
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

        def calculate_probas(
            voter, weights=None, verbose=None
        ):

            ensemble_proba = 0

            n_iter = len(voter)

            assert (
                n_iter > 0
            ), "no estimator found in `RandomBag` ensemble"

            if weights is None:

                for idx, elt in voter.items():

                    ensemble_proba += elt.predict_proba(X)

                    if verbose == 1:
                        pbar.update(idx)

                if verbose == 1:
                    pbar.update(n_iter)

                return ensemble_proba / n_iter

            # if weights is not None:
            for idx, elt in voter.items():

                ensemble_proba += weights[
                    idx
                ] * elt.predict_proba(X)

                if verbose == 1:
                    pbar.update(idx)

            if verbose == 1:
                pbar.update(n_iter)

            return ensemble_proba

        # end calculate_probas ----

        if self.n_jobs is None:

            if self.verbose == 1:
                pbar = Progbar(self.n_estimators)

            if weights is None:

                return calculate_probas(
                    self.voter, verbose=self.verbose
                )

            # if weights is not None:
            self.weights = weights

            return calculate_probas(
                self.voter, weights, verbose=self.verbose
            )

        # if self.n_jobs is not None:
        def predict_estimator(m):
            return self.voter[m].predict_proba(X)

        if self.verbose == 1:

            preds = Parallel(
                n_jobs=self.n_jobs, prefer="threads"
            )(
                delayed(predict_estimator)(m)
                for m in tqdm(range(self.n_estimators))
            )

        else:

            preds = Parallel(
                n_jobs=self.n_jobs, prefer="threads"
            )(
                delayed(predict_estimator)(m)
                for m in range(self.n_estimators)
            )

        ensemble_proba = 0

        if weights is None:

            for i in range(self.n_estimators):

                ensemble_proba += preds[i]

            return ensemble_proba / self.n_estimators

        for i in range(self.n_estimators):

            ensemble_proba += weights[i] * preds[i]

        return ensemble_proba

    def score(
        self, X, y, weights=None, scoring=None, **kwargs
    ):
        """ Score the model on test set covariates X and response y. """

        preds = self.predict(X, weights, **kwargs)

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
