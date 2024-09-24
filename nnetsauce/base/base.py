# Authors: Thierry Moudiki
#
# License: BSD 3 Clear

import copy
import numpy as np
import pandas as pd
import platform
import warnings
import sklearn.metrics as skm

from functools import partial
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from ..utils import activations as ac
from ..utils import matrixops as mo
from ..utils import misc as mx
from ..simulation import (
    generate_sobol,
    generate_uniform,
    generate_hammersley,
    generate_halton,
)
from ..sampling import SubSampler


try:
    import jax.nn as jnn
    import jax.numpy as jnp
except ImportError:
    pass


class Base(BaseEstimator):
    """Base model from which all the other classes inherit.

    This class contains the most important data preprocessing/feature engineering methods.

    Parameters:

        n_hidden_features: int
            number of nodes in the hidden layer

        activation_name: str
            activation function: 'relu', 'tanh', 'sigmoid', 'prelu' or 'elu'

        a: float
            hyperparameter for 'prelu' or 'elu' activation function

        nodes_sim: str
            type of simulation for hidden layer nodes: 'sobol', 'hammersley', 'halton',
            'uniform'

        bias: boolean
            indicates if the hidden layer contains a bias term (True) or
            not (False)

        dropout: float
            regularization parameter; (random) percentage of nodes dropped out
            of the training

        direct_link: boolean
            indicates if the original features are included (True) in model's
            fitting or not (False)

        n_clusters: int
            number of clusters for type_clust='kmeans' or type_clust='gmm'
            clustering (could be 0: no clustering)

        cluster_encode: bool
            defines how the variable containing clusters is treated (default is one-hot);
            if `False`, then labels are used, without one-hot encoding

        type_clust: str
            type of clustering method: currently k-means ('kmeans') or Gaussian
            Mixture Model ('gmm')

        type_scaling: a tuple of 3 strings
            scaling methods for inputs, hidden layer, and clustering respectively
            (and when relevant).
            Currently available: standardization ('std') or MinMax scaling ('minmax') or robust scaling ('robust') or  max absolute scaling ('maxabs')

        col_sample: float
            percentage of features randomly chosen for training

        row_sample: float
            percentage of rows chosen for training, by stratified bootstrapping

        seed: int
            reproducibility seed for nodes_sim=='uniform', clustering and dropout

        backend: str
            "cpu" or "gpu" or "tpu"

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
        cluster_encode=True,
        type_clust="kmeans",
        type_scaling=("std", "std", "std"),
        col_sample=1,
        row_sample=1,
        seed=123,
        backend="cpu",
    ):
        # input checks -----

        sys_platform = platform.system()

        if (sys_platform == "Windows") and (backend in ("gpu", "tpu")):
            warnings.warn(
                "No GPU/TPU computing on Windows yet, backend set to 'cpu'"
            )
            backend = "cpu"

        assert activation_name in (
            "relu",
            "tanh",
            "sigmoid",
            "prelu",
            "elu",
        ), "'activation_name' must be in ('relu', 'tanh', 'sigmoid','prelu', 'elu')"

        assert nodes_sim in (
            "sobol",
            "hammersley",
            "uniform",
            "halton",
        ), "'nodes_sim' must be in ('sobol', 'hammersley', 'uniform', 'halton')"

        assert type_clust in (
            "kmeans",
            "gmm",
        ), "'type_clust' must be in ('kmeans', 'gmm')"

        assert (len(type_scaling) == 3) & all(
            type_scaling[i] in ("minmax", "std", "robust", "maxabs")
            for i in range(len(type_scaling))
        ), "'type_scaling' must have length 3, and available scaling methods are 'minmax' scaling, standardization ('std'), robust scaling ('robust') and max absolute ('maxabs')"

        assert (col_sample >= 0) & (
            col_sample <= 1
        ), "'col_sample' must be comprised between 0 and 1 (both included)"

        assert backend in (
            "cpu",
            "gpu",
            "tpu",
        ), "must have 'backend' in ('cpu', 'gpu', 'tpu')"

        self.n_hidden_features = n_hidden_features
        self.activation_name = activation_name
        self.a = a
        self.nodes_sim = nodes_sim
        self.bias = bias
        self.seed = seed
        self.backend = backend
        self.dropout = dropout
        self.direct_link = direct_link
        self.cluster_encode = cluster_encode
        self.type_clust = type_clust
        self.type_scaling = type_scaling
        self.col_sample = col_sample
        self.row_sample = row_sample
        self.n_clusters = n_clusters
        if isinstance(self, RegressorMixin):
            self.type_fit = "regression"
        elif isinstance(self, ClassifierMixin):
            self.type_fit = "classification"
        self.subsampler_ = None
        self.index_col_ = None
        self.index_row_ = True
        self.clustering_obj_ = None
        self.clustering_scaler_ = None
        self.nn_scaler_ = None
        self.scaler_ = None
        self.encoder_ = None
        self.W_ = None
        self.X_ = None
        self.y_ = None
        self.y_mean_ = None
        self.beta_ = None

        # activation function -----
        if sys_platform in ("Linux", "Darwin"):
            activation_options = {
                "relu": ac.relu if (self.backend == "cpu") else jnn.relu,
                "tanh": np.tanh if (self.backend == "cpu") else jnp.tanh,
                "sigmoid": (
                    ac.sigmoid if (self.backend == "cpu") else jnn.sigmoid
                ),
                "prelu": partial(ac.prelu, a=a),
                "elu": (
                    partial(ac.elu, a=a)
                    if (self.backend == "cpu")
                    else partial(jnn.elu, a=a)
                ),
            }
        else:  # on Windows currently, no JAX
            activation_options = {
                "relu": (
                    ac.relu if (self.backend == "cpu") else NotImplementedError
                ),
                "tanh": (
                    np.tanh if (self.backend == "cpu") else NotImplementedError
                ),
                "sigmoid": (
                    ac.sigmoid
                    if (self.backend == "cpu")
                    else NotImplementedError
                ),
                "prelu": partial(ac.prelu, a=a),
                "elu": (
                    partial(ac.elu, a=a)
                    if (self.backend == "cpu")
                    else NotImplementedError
                ),
            }
        self.activation_func = activation_options[activation_name]

    # "preprocessing" methods to be inherited -----

    def encode_clusters(self, X=None, predict=False, **kwargs):  #
        """Create new covariates with kmeans or GMM clustering

        Parameters:

            X: {array-like}, shape = [n_samples, n_features]
                Training vectors, where n_samples is the number
                of samples and n_features is the number of features.

            predict: boolean
                is False on training set and True on test set

            **kwargs:
                additional parameters to be passed to the
                clustering method

        Returns:

            Clusters' matrix, one-hot encoded: {array-like}

        """

        np.random.seed(self.seed)

        if X is None:
            X = self.X_

        if isinstance(X, pd.DataFrame):
            X = copy.deepcopy(X.values.astype(float))

        if predict is False:  # encode training set
            # scale input data before clustering
            self.clustering_scaler_, scaled_X = mo.scale_covariates(
                X, choice=self.type_scaling[2]
            )

            self.clustering_obj_, X_clustered = mo.cluster_covariates(
                scaled_X,
                self.n_clusters,
                self.seed,
                type_clust=self.type_clust,
                **kwargs
            )

            if self.cluster_encode == True:
                return mo.one_hot_encode(X_clustered, self.n_clusters).astype(
                    np.float16
                )

            return X_clustered.astype(np.float16)

        # if predict == True, encode test set
        X_clustered = self.clustering_obj_.predict(
            self.clustering_scaler_.transform(X)
        )

        if self.cluster_encode == True:
            return mo.one_hot_encode(X_clustered, self.n_clusters).astype(
                np.float16
            )

        return X_clustered.astype(np.float16)

    def create_layer(self, scaled_X, W=None):
        """Create hidden layer.

        Parameters:

            scaled_X: {array-like}, shape = [n_samples, n_features]
                Training vectors, where n_samples is the number
                of samples and n_features is the number of features

            W: {array-like}, shape = [n_features, hidden_features]
                if provided, constructs the hidden layer with W; otherwise computed internally

        Returns:

            Hidden layer matrix: {array-like}

        """

        n_features = scaled_X.shape[1]

        # hash_sim = {
        #         "sobol": generate_sobol,
        #         "hammersley": generate_hammersley,
        #         "uniform": generate_uniform,
        #         "halton": generate_halton
        #     }

        if self.bias is False:  # no bias term in the hidden layer
            if W is None:
                if self.nodes_sim == "sobol":
                    self.W_ = generate_sobol(
                        n_dims=n_features,
                        n_points=self.n_hidden_features,
                        seed=self.seed,
                    )
                elif self.nodes_sim == "hammersley":
                    self.W_ = generate_hammersley(
                        n_dims=n_features,
                        n_points=self.n_hidden_features,
                        seed=self.seed,
                    )
                elif self.nodes_sim == "uniform":
                    self.W_ = generate_uniform(
                        n_dims=n_features,
                        n_points=self.n_hidden_features,
                        seed=self.seed,
                    )
                else:
                    self.W_ = generate_halton(
                        n_dims=n_features,
                        n_points=self.n_hidden_features,
                        seed=self.seed,
                    )

                # self.W_ = hash_sim[self.nodes_sim](
                #             n_dims=n_features,
                #             n_points=self.n_hidden_features,
                #             seed=self.seed,
                #         )

                assert (
                    scaled_X.shape[1] == self.W_.shape[0]
                ), "check dimensions of covariates X and matrix W"

                return mo.dropout(
                    x=self.activation_func(
                        mo.safe_sparse_dot(
                            a=scaled_X, b=self.W_, backend=self.backend
                        )
                    ),
                    drop_prob=self.dropout,
                    seed=self.seed,
                )

            # W is not none
            assert (
                scaled_X.shape[1] == W.shape[0]
            ), "check dimensions of covariates X and matrix W"

            # self.W_ = W
            return mo.dropout(
                x=self.activation_func(
                    mo.safe_sparse_dot(a=scaled_X, b=W, backend=self.backend)
                ),
                drop_prob=self.dropout,
                seed=self.seed,
            )

        # with bias term in the hidden layer
        if W is None:
            n_features_1 = n_features + 1

            if self.nodes_sim == "sobol":
                self.W_ = generate_sobol(
                    n_dims=n_features_1,
                    n_points=self.n_hidden_features,
                    seed=self.seed,
                )
            elif self.nodes_sim == "hammersley":
                self.W_ = generate_hammersley(
                    n_dims=n_features_1,
                    n_points=self.n_hidden_features,
                    seed=self.seed,
                )
            elif self.nodes_sim == "uniform":
                self.W_ = generate_uniform(
                    n_dims=n_features_1,
                    n_points=self.n_hidden_features,
                    seed=self.seed,
                )
            else:
                self.W_ = generate_halton(
                    n_dims=n_features_1,
                    n_points=self.n_hidden_features,
                    seed=self.seed,
                )

            # self.W_ = hash_sim[self.nodes_sim](
            #         n_dims=n_features_1,
            #         n_points=self.n_hidden_features,
            #         seed=self.seed,
            #     )

            return mo.dropout(
                x=self.activation_func(
                    mo.safe_sparse_dot(
                        a=mo.cbind(
                            np.ones(scaled_X.shape[0]),
                            scaled_X,
                            backend=self.backend,
                        ),
                        b=self.W_,
                        backend=self.backend,
                    )
                ),
                drop_prob=self.dropout,
                seed=self.seed,
            )

        # W is not None
        # self.W_ = W
        return mo.dropout(
            x=self.activation_func(
                mo.safe_sparse_dot(
                    a=mo.cbind(
                        np.ones(scaled_X.shape[0]),
                        scaled_X,
                        backend=self.backend,
                    ),
                    b=W,
                    backend=self.backend,
                )
            ),
            drop_prob=self.dropout,
            seed=self.seed,
        )

    def cook_training_set(self, y=None, X=None, W=None, **kwargs):
        """Create new hidden features for training set, with hidden layer, center the response.

        Parameters:

            y: array-like, shape = [n_samples]
                Target values

            X: {array-like}, shape = [n_samples, n_features]
                Training vectors, where n_samples is the number
                of samples and n_features is the number of features

            W: {array-like}, shape = [n_features, hidden_features]
                if provided, constructs the hidden layer via W

        Returns:

            (centered response, direct link + hidden layer matrix): {tuple}

        """

        # either X and y are stored or not
        # assert ((y is None) & (X is None)) | ((y is not None) & (X is not None))
        if self.n_hidden_features > 0:  # has a hidden layer
            assert (
                len(self.type_scaling) >= 2
            ), "must have len(self.type_scaling) >= 2 when self.n_hidden_features > 0"

        if X is None:
            if self.col_sample == 1:
                input_X = self.X_
            else:
                n_features = self.X_.shape[1]
                new_n_features = int(np.ceil(n_features * self.col_sample))
                assert (
                    new_n_features >= 1
                ), "check class attribute 'col_sample' and the number of covariates provided for X"
                np.random.seed(self.seed)
                index_col = np.random.choice(
                    range(n_features), size=new_n_features, replace=False
                )
                self.index_col_ = index_col
                input_X = self.X_[:, self.index_col_]

        else:  # X is not None # keep X vs self.X_
            if isinstance(X, pd.DataFrame):
                X = copy.deepcopy(X.values.astype(float))

            if self.col_sample == 1:
                input_X = X
            else:
                n_features = X.shape[1]
                new_n_features = int(np.ceil(n_features * self.col_sample))
                assert (
                    new_n_features >= 1
                ), "check class attribute 'col_sample' and the number of covariates provided for X"
                np.random.seed(self.seed)
                index_col = np.random.choice(
                    range(n_features), size=new_n_features, replace=False
                )
                self.index_col_ = index_col
                input_X = X[:, self.index_col_]

        if (
            self.n_clusters <= 0
        ):  # data without any clustering: self.n_clusters is None -----
            if self.n_hidden_features > 0:  # with hidden layer
                self.nn_scaler_, scaled_X = mo.scale_covariates(
                    input_X, choice=self.type_scaling[1]
                )
                Phi_X = (
                    self.create_layer(scaled_X)
                    if W is None
                    else self.create_layer(scaled_X, W=W)
                )
                Z = (
                    mo.cbind(input_X, Phi_X, backend=self.backend)
                    if self.direct_link is True
                    else Phi_X
                )
                self.scaler_, scaled_Z = mo.scale_covariates(
                    Z, choice=self.type_scaling[0]
                )
            else:  # no hidden layer
                Z = input_X
                self.scaler_, scaled_Z = mo.scale_covariates(
                    Z, choice=self.type_scaling[0]
                )
        else:  # data with clustering: self.n_clusters is not None ----- # keep
            augmented_X = mo.cbind(
                input_X,
                self.encode_clusters(input_X, **kwargs),
                backend=self.backend,
            )

            if self.n_hidden_features > 0:  # with hidden layer
                self.nn_scaler_, scaled_X = mo.scale_covariates(
                    augmented_X, choice=self.type_scaling[1]
                )
                Phi_X = (
                    self.create_layer(scaled_X)
                    if W is None
                    else self.create_layer(scaled_X, W=W)
                )
                Z = (
                    mo.cbind(augmented_X, Phi_X, backend=self.backend)
                    if self.direct_link is True
                    else Phi_X
                )
                self.scaler_, scaled_Z = mo.scale_covariates(
                    Z, choice=self.type_scaling[0]
                )
            else:  # no hidden layer
                Z = augmented_X
                self.scaler_, scaled_Z = mo.scale_covariates(
                    Z, choice=self.type_scaling[0]
                )

        # Returning model inputs -----        
        if mx.is_factor(y) is False:  # regression
            # center y
            if y is None:
                self.y_mean_, centered_y = mo.center_response(self.y_)
            else:
                self.y_mean_, centered_y = mo.center_response(y)

            # y is subsampled
            if self.row_sample < 1:
                n, p = Z.shape

                self.subsampler_ = (
                    SubSampler(
                        y=self.y_, row_sample=self.row_sample, seed=self.seed
                    )
                    if y is None
                    else SubSampler(
                        y=y, row_sample=self.row_sample, seed=self.seed
                    )
                )

                self.index_row_ = self.subsampler_.subsample()

                n_row_sample = len(self.index_row_)
                # regression
                return (
                    centered_y[self.index_row_].reshape(n_row_sample),
                    self.scaler_.transform(
                        Z[self.index_row_, :].reshape(n_row_sample, p)
                    ),
                )
            # y is not subsampled
            # regression
            return (centered_y, self.scaler_.transform(Z))

        # classification
        # y is subsampled
        if self.row_sample < 1:
            n, p = Z.shape

            self.subsampler_ = (
                SubSampler(
                    y=self.y_, row_sample=self.row_sample, seed=self.seed
                )
                if y is None
                else SubSampler(y=y, row_sample=self.row_sample, seed=self.seed)
            )

            self.index_row_ = self.subsampler_.subsample()

            n_row_sample = len(self.index_row_)
            # classification
            return (
                y[self.index_row_].reshape(n_row_sample),
                self.scaler_.transform(
                    Z[self.index_row_, :].reshape(n_row_sample, p)
                ),
            )
        # y is not subsampled
        # classification
        return (y, self.scaler_.transform(Z))

    def cook_test_set(self, X, **kwargs):
        """Transform data from test set, with hidden layer.

        Parameters:

            X: {array-like}, shape = [n_samples, n_features]
                Training vectors, where n_samples is the number
                of samples and n_features is the number of features

            **kwargs: additional parameters to be passed to self.encode_cluster

        Returns:

            Transformed test set : {array-like}
        """

        if isinstance(X, pd.DataFrame):
            X = copy.deepcopy(X.values.astype(float))

        if (
            self.n_clusters == 0
        ):  # data without clustering: self.n_clusters is None -----
            if self.n_hidden_features > 0:
                # if hidden layer
                scaled_X = (
                    self.nn_scaler_.transform(X)
                    if (self.col_sample == 1)
                    else self.nn_scaler_.transform(X[:, self.index_col_])
                )
                Phi_X = self.create_layer(scaled_X, self.W_)
                if self.direct_link == True:
                    return self.scaler_.transform(
                        mo.cbind(scaled_X, Phi_X, backend=self.backend)
                    )
                # when self.direct_link == False
                return self.scaler_.transform(Phi_X)
            # if no hidden layer # self.n_hidden_features == 0
            return self.scaler_.transform(X)

        # data with clustering: self.n_clusters > 0 -----
        if self.col_sample == 1:
            predicted_clusters = self.encode_clusters(
                X=X, predict=True, **kwargs
            )
            augmented_X = mo.cbind(X, predicted_clusters, backend=self.backend)
        else:
            predicted_clusters = self.encode_clusters(
                X=X[:, self.index_col_], predict=True, **kwargs
            )
            augmented_X = mo.cbind(
                X[:, self.index_col_], predicted_clusters, backend=self.backend
            )

        if self.n_hidden_features > 0:  # if hidden layer
            scaled_X = self.nn_scaler_.transform(augmented_X)
            Phi_X = self.create_layer(scaled_X, self.W_)
            if self.direct_link == True:
                return self.scaler_.transform(
                    mo.cbind(augmented_X, Phi_X, backend=self.backend)
                )
            return self.scaler_.transform(Phi_X)

        # if no hidden layer
        return self.scaler_.transform(augmented_X)

    def score(self, X, y, scoring=None, **kwargs):
        """Score the model on test set features X and response y.

        Parameters:

            X: {array-like}, shape = [n_samples, n_features]
                Training vectors, where n_samples is the number
                of samples and n_features is the number of features

            y: array-like, shape = [n_samples]
                Target values

            scoring: str
                must be in ('explained_variance', 'neg_mean_absolute_error',
                            'neg_mean_squared_error', 'neg_mean_squared_log_error',
                            'neg_median_absolute_error', 'r2')

            **kwargs: additional parameters to be passed to scoring functions

        Returns:

        model scores: {array-like}

        """

        preds = self.predict(X)

        if self.type_fit == "classification":

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
                "accuracy": skm.accuracy_score,
                "average_precision": skm.average_precision_score,
                "brier_score_loss": skm.brier_score_loss,
                "f1": skm.f1_score,
                "f1_micro": skm.f1_score,
                "f1_macro": skm.f1_score,
                "f1_weighted": skm.f1_score,
                "f1_samples": skm.f1_score,
                "neg_log_loss": skm.log_loss,
                "precision": skm.precision_score,
                "recall": skm.recall_score,
                "roc_auc": skm.roc_auc_score,
            }

            return scoring_options[scoring](y, preds, **kwargs)

        if self.type_fit == "regression":

            if (
                type(preds) == tuple
            ):  # if there are std. devs in the predictions
                preds = preds[0]

            if scoring is None:
                scoring = "neg_root_mean_squared_error"

            # check inputs
            assert scoring in (
                "explained_variance",
                "neg_mean_absolute_error",
                "neg_mean_squared_error",
                "neg_mean_squared_log_error",
                "neg_median_absolute_error",
                "neg_root_mean_squared_error",
                "r2",
            ), "'scoring' should be in ('explained_variance', 'neg_mean_absolute_error', \
                            'neg_mean_squared_error', 'neg_mean_squared_log_error', \
                            'neg_median_absolute_error', 'r2', 'neg_root_mean_squared_error')"

            scoring_options = {
                "neg_root_mean_squared_error": skm.root_mean_squared_error,
                "explained_variance": skm.explained_variance_score,
                "neg_mean_absolute_error": skm.median_absolute_error,
                "neg_mean_squared_error": skm.mean_squared_error,
                "neg_mean_squared_log_error": skm.mean_squared_log_error,
                "neg_median_absolute_error": skm.median_absolute_error,
                "r2": skm.r2_score,
            }

            return scoring_options[scoring](y, preds, **kwargs)
