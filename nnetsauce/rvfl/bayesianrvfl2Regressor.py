# Authors: Thierry Moudiki
#
# License: BSD 3

import numpy as np
import sklearn.metrics as skm2
from ..base import Base
from ..utils import misc as mx
from ..utils import matrixops as mo
from ..utils import lmfuncs as lmf
from sklearn.base import RegressorMixin


class BayesianRVFL2Regressor(Base, RegressorMixin):
    """Bayesian Random Vector Functional Link Network regression with two priors

    Parameters:

        n_hidden_features: int
            number of nodes in the hidden layer

        activation_name: str
            activation function: 'relu', 'tanh', 'sigmoid', 'prelu' or 'elu'

        a: float
            hyperparameter for 'prelu' or 'elu' activation function

        nodes_sim: str
            type of simulation for the nodes: 'sobol', 'hammersley', 'halton', 'uniform'

        bias: boolean
            indicates if the hidden layer contains a bias term (True) or not (False)

        dropout: float
            regularization parameter; (random) percentage of nodes dropped out
            of the training

        direct_link: boolean
            indicates if the original features are included (True) in model''s fitting or not (False)

        n_clusters: int
            number of clusters for 'kmeans' or 'gmm' clustering (could be 0: no clustering)

        cluster_encode: bool
            defines how the variable containing clusters is treated (default is one-hot)
            if `False`, then labels are used, without one-hot encoding

        type_clust: str
            type of clustering method: currently k-means ('kmeans') or Gaussian Mixture Model ('gmm')

        type_scaling: a tuple of 3 strings
            scaling methods for inputs, hidden layer, and clustering respectively
            (and when relevant).
            Currently available: standardization ('std') or MinMax scaling ('minmax')

        seed: int
            reproducibility seed for nodes_sim=='uniform'

        s1: float
            std. dev. of init. regression parameters in Bayesian Ridge Regression

        s2: float
            std. dev. of augmented regression parameters in Bayesian Ridge Regression

        sigma: float
            std. dev. of residuals in Bayesian Ridge Regression

        return_std: boolean
            if True, uncertainty around predictions is evaluated

        backend: str
            "cpu" or "gpu" or "tpu"

    Attributes:

        beta_: array-like
            regression''s coefficients

        Sigma_: array-like
            covariance of the distribution of fitted parameters

        GCV_: float
            Generalized cross-validation error

        y_mean_: float
            average response    

    Examples:

    ```python
    TBD
    ```

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
        n_clusters=0,
        cluster_encode=True,
        type_clust="kmeans",
        type_scaling=("std", "std", "std"),
        seed=123,
        s1=0.1,
        s2=0.1,
        sigma=0.05,
        return_std=True,
        backend="cpu",
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
            cluster_encode=cluster_encode,
            type_clust=type_clust,
            type_scaling=type_scaling,
            seed=seed,
            backend=backend,
        )

        self.s1 = s1
        self.s2 = s2
        self.sigma = sigma
        self.beta_ = None
        self.Sigma_ = None
        self.GCV_ = None
        self.return_std = return_std

    def fit(self, X, y, **kwargs):
        """Fit BayesianRVFL2Regressor to training data (X, y)

        Parameters:

            X: {array-like}, shape = [n_samples, n_features]
                Training vectors, where n_samples is the number
                of samples and n_features is the number of features

            y: array-like, shape = [n_samples]
                Target values

            **kwargs: additional parameters to be passed to
                    self.cook_training_set

        Returns:

            self: object

        """

        centered_y, scaled_Z = self.cook_training_set(y=y, X=X, **kwargs)

        n, p = X.shape
        q = self.n_hidden_features

        if self.direct_link == True:

            r = p + self.n_clusters

            block11 = (self.s1 ** 2) * np.eye(r)
            block12 = np.zeros((r, q))
            block21 = np.zeros((q, r))
            block22 = (self.s2 ** 2) * np.eye(q)

            Sigma_prior = mo.rbind(
                x=mo.cbind(x=block11, y=block12, backend=self.backend),
                y=mo.cbind(x=block21, y=block22, backend=self.backend),
                backend=self.backend,
            )

        else:

            Sigma_prior = (self.s2 ** 2) * np.eye(q)

        fit_obj = lmf.beta_Sigma_hat_rvfl2(
            X=scaled_Z,
            y=centered_y,
            Sigma=Sigma_prior,
            sigma=self.sigma,
            fit_intercept=False,
            return_cov=self.return_std,
            backend=self.backend,
        )

        self.beta_ = fit_obj["beta_hat"]

        if self.return_std == True:
            self.Sigma_ = fit_obj["Sigma_hat"]

        self.GCV_ = fit_obj["GCV"]

        return self

    def predict(self, X, return_std=False, **kwargs):
        """Predict test data X.

        Parameters:

            X: {array-like}, shape = [n_samples, n_features]
                Training vectors, where n_samples is the number
                of samples and n_features is the number of features.

            return_std: {boolean}, standard dev. is returned or not

            **kwargs: additional parameters to be passed to
                    self.cook_test_set

        Returns:

            model predictions: {array-like}

        """

        if len(X.shape) == 1:  # one observation in the test set only
            n_features = X.shape[0]
            new_X = mo.rbind(
                x=X.reshape(1, n_features),
                y=np.ones(n_features).reshape(1, n_features),
                backend=self.backend,
            )

        self.return_std = return_std

        if self.return_std == False:

            if len(X.shape) == 1:

                return (
                    self.y_mean_
                    + mo.safe_sparse_dot(
                        self.cook_test_set(new_X, **kwargs),
                        self.beta_,
                        backend=self.backend,
                    )
                )[0]

            return self.y_mean_ + mo.safe_sparse_dot(
                self.cook_test_set(X, **kwargs), self.beta_, backend=self.backend
            )

        else:  # confidence interval required for preds?

            if len(X.shape) == 1:

                Z = self.cook_test_set(new_X, **kwargs)

                pred_obj = lmf.beta_Sigma_hat_rvfl2(
                    X_star=Z,
                    return_cov=self.return_std,
                    beta_hat_=self.beta_,
                    Sigma_hat_=self.Sigma_,
                    backend=self.backend,
                )

                return (
                    self.y_mean_ + pred_obj["preds"][0],
                    pred_obj["preds_std"][0],
                )

            Z = self.cook_test_set(X, **kwargs)

            pred_obj = lmf.beta_Sigma_hat_rvfl2(
                X_star=Z,
                return_cov=self.return_std,
                beta_hat_=self.beta_,
                Sigma_hat_=self.Sigma_,
                backend=self.backend,
            )

            return (self.y_mean_ + pred_obj["preds"], pred_obj["preds_std"])

    def score(self, X, y, scoring=None, **kwargs):
        """ Score the model on test set features X and response y. 

        Parameters:
        
            X: {array-like}, shape = [n_samples, n_features]
                Training vectors, where n_samples is the number 
                of samples and n_features is the number of features

            y: array-like, shape = [n_samples]
                Target values

            scoring: str
                must be in ('explained_variance', 'neg_mean_absolute_error', \
                            'neg_mean_squared_error', 'neg_mean_squared_log_error', \
                            'neg_median_absolute_error', 'r2')
            
            **kwargs: additional parameters to be passed to scoring functions
                
        Returns: 
        
            model scores: {array-like}
            
        """

        preds = self.predict(X)

        if self.return_std:  # if there are std. devs in the predictions
            preds = preds[0]

        if mx.is_factor(y):  # classification

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

        else:  # regression

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
                "explained_variance": skm2.explained_variance_score,
                "neg_mean_absolute_error": skm2.mean_absolute_error,
                "neg_mean_squared_error": skm2.mean_squared_error,
                "neg_mean_squared_log_error": skm2.mean_squared_log_error,
                "neg_median_absolute_error": skm2.median_absolute_error,
                "r2": skm2.r2_score,
            }

        return scoring_options[scoring](y, preds, **kwargs)
