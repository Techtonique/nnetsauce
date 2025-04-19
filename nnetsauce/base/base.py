# Authors: Thierry Moudiki
#
# License: BSD 3 Clear

import copy
import numpy as np
import pandas as pd
import platform
import warnings
import sklearn.metrics as skm

from collections import namedtuple
from functools import partial
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
)
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
            warnings.warn("No GPU/TPU computing on Windows yet, backend set to 'cpu'")
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
                "sigmoid": (ac.sigmoid if (self.backend == "cpu") else jnn.sigmoid),
                "prelu": partial(ac.prelu, a=a),
                "elu": (
                    partial(ac.elu, a=a)
                    if (self.backend == "cpu")
                    else partial(jnn.elu, a=a)
                ),
            }
        else:  # on Windows currently, no JAX
            activation_options = {
                "relu": (ac.relu if (self.backend == "cpu") else NotImplementedError),
                "tanh": (np.tanh if (self.backend == "cpu") else NotImplementedError),
                "sigmoid": (
                    ac.sigmoid if (self.backend == "cpu") else NotImplementedError
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

    def encode_clusters(self, X=None, predict=False, scaler=None, **kwargs):  #
        """Create new covariates with kmeans or GMM clustering

        Parameters:

            X: {array-like}, shape = [n_samples, n_features]
                Training vectors, where n_samples is the number
                of samples and n_features is the number of features.

            predict: boolean
                is False on training set and True on test set

            scaler: {object} of class StandardScaler, MinMaxScaler, RobustScaler or MaxAbsScaler
                if scaler has already been fitted on training data (online training), it can be passed here

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

        if len(X.shape) == 1:
            X = X.reshape(1, -1)

        if predict is False:  # encode training set

            # scale input data before clustering
            self.clustering_scaler_, scaled_X = mo.scale_covariates(
                X, choice=self.type_scaling[2], scaler=self.clustering_scaler_
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
        X_clustered = self.clustering_obj_.predict(self.clustering_scaler_.transform(X))

        if self.cluster_encode == True:
            return mo.one_hot_encode(X_clustered, self.n_clusters).astype(np.float16)

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
                        mo.safe_sparse_dot(a=scaled_X, b=self.W_, backend=self.backend)
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

        if self.n_clusters <= 0:
            # data without any clustering: self.n_clusters is None -----

            if self.n_hidden_features > 0:  # with hidden layer

                self.nn_scaler_, scaled_X = mo.scale_covariates(
                    input_X, choice=self.type_scaling[1], scaler=self.nn_scaler_
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
                    Z, choice=self.type_scaling[0], scaler=self.scaler_
                )
            else:  # no hidden layer
                Z = input_X
                self.scaler_, scaled_Z = mo.scale_covariates(
                    Z, choice=self.type_scaling[0], scaler=self.scaler_
                )

        else:

            # data with clustering: self.n_clusters is not None ----- # keep

            augmented_X = mo.cbind(
                input_X,
                self.encode_clusters(input_X, **kwargs),
                backend=self.backend,
            )

            if self.n_hidden_features > 0:  # with hidden layer

                self.nn_scaler_, scaled_X = mo.scale_covariates(
                    augmented_X,
                    choice=self.type_scaling[1],
                    scaler=self.nn_scaler_,
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
                    Z, choice=self.type_scaling[0], scaler=self.scaler_
                )
            else:  # no hidden layer
                Z = augmented_X
                self.scaler_, scaled_Z = mo.scale_covariates(
                    Z, choice=self.type_scaling[0], scaler=self.scaler_
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
                    SubSampler(y=self.y_, row_sample=self.row_sample, seed=self.seed)
                    if y is None
                    else SubSampler(y=y, row_sample=self.row_sample, seed=self.seed)
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
                SubSampler(y=self.y_, row_sample=self.row_sample, seed=self.seed)
                if y is None
                else SubSampler(y=y, row_sample=self.row_sample, seed=self.seed)
            )

            self.index_row_ = self.subsampler_.subsample()

            n_row_sample = len(self.index_row_)
            # classification
            return (
                y[self.index_row_].reshape(n_row_sample),
                self.scaler_.transform(Z[self.index_row_, :].reshape(n_row_sample, p)),
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

        if len(X.shape) == 1:
            X = X.reshape(1, -1)

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
            predicted_clusters = self.encode_clusters(X=X, predict=True, **kwargs)
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

    def cross_val_score(
        self,
        X,
        y,
        cv=5,
        scoring="accuracy",
        random_state=42,
        n_jobs=-1,
        epsilon=0.5,
        penalized=True,
        objective="abs",
        **kwargs
    ):
        """
        Penalized Cross-validation score for a model.

        Parameters:

            X: {array-like}, shape = [n_samples, n_features]
                Training vectors, where n_samples is the number
                of samples and n_features is the number of features

            y: array-like, shape = [n_samples]
                Target values

            X_test: {array-like}, shape = [n_samples, n_features]
                Test vectors, where n_samples is the number
                of samples and n_features is the number of features

            y_test: array-like, shape = [n_samples]
                Target values

            cv: int
                Number of folds

            scoring: str
                Scoring metric

            random_state: int
                Random state

            n_jobs: int
                Number of jobs to run in parallel

            epsilon: float
                Penalty parameter

            penalized: bool
                Whether to obtain penalized cross-validation score or not

            objective: str
                'abs': Minimize the absolute difference between cross-validation score and validation score
                'relative': Minimize the relative difference between cross-validation score and validation score
        Returns:

            A namedtuple with the following fields:
                - cv_score: float
                    cross-validation score
                - val_score: float
                    validation score
                - penalized_score: float
                    penalized cross-validation score: cv_score / val_score + epsilon*(1/val_score + 1/cv_score)
                    If higher scoring metric is better, minimize the function result.
                    If lower scoring metric is better, maximize the function result.
        """
        if scoring == "accuracy":
            scoring_func = accuracy_score
        elif scoring == "balanced_accuracy":
            scoring_func = balanced_accuracy_score
        elif scoring == "f1":
            scoring_func = f1_score
        elif scoring == "roc_auc":
            scoring_func = roc_auc_score
        elif scoring == "r2":
            scoring_func = r2_score
        elif scoring == "mse":
            scoring_func = mean_squared_error
        elif scoring == "mae":
            scoring_func = mean_absolute_error
        elif scoring == "mape":
            scoring_func = mean_absolute_percentage_error
        elif scoring == "rmse":

            def scoring_func(y_true, y_pred):
                return np.sqrt(mean_squared_error(y_true, y_pred))

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=random_state
        )

        res = cross_val_score(
            self, X_train, y_train, cv=cv, scoring=scoring, n_jobs=n_jobs
        )  # cross-validation error

        if penalized == False:
            return res

        DescribeResult = namedtuple(
            "DescribeResult", ["cv_score", "val_score", "penalized_score"]
        )

        numerator = res.mean()

        # Evaluate on the (cv+1)-th fold
        preds_val = self.fit(X_train, y_train).predict(X_val)
        try:
            denominator = scoring(y_val, preds_val)  # validation error
        except Exception as e:
            denominator = scoring_func(y_val, preds_val)

        # if higher is better
        if objective == "abs":
            penalized_score = np.abs(numerator - denominator) + epsilon * (
                1 / denominator + 1 / numerator
            )
        elif objective == "relative":
            ratio = numerator / denominator
            penalized_score = np.abs(ratio - 1) + epsilon * (
                1 / denominator + 1 / numerator
            )

        return DescribeResult(
            cv_score=numerator,
            val_score=denominator,
            penalized_score=penalized_score,
        )


class RidgeRegressor(Base, RegressorMixin):
    """Ridge Regression model with JAX support.

    Parameters:
        lambda_: float or array-like
            Ridge regularization parameter(s). Default is 0.

        All other parameters are inherited from Base class.
    """

    def __init__(
        self,
        lambda_=0.0,
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
            col_sample=col_sample,
            row_sample=row_sample,
            seed=seed,
            backend=backend,
        )
        self.lambda_ = lambda_

    def _center_scale_xy(self, X, y):
        """Center X and y, scale X."""
        n = X.shape[0]

        # Center X and y
        X_mean = jnp.mean(X, axis=0)
        y_mean = jnp.mean(y)
        X_centered = X - X_mean
        y_centered = y - y_mean

        # Scale X
        X_scale = jnp.sqrt(jnp.sum(X_centered**2, axis=0) / n)
        X_scaled = X_centered / X_scale

        return X_scaled, y_centered, X_mean, y_mean, X_scale

    def fit(self, X, y, **kwargs):
        """Fit Ridge regression model.

        Parameters:
            X : array-like of shape (n_samples, n_features)
                Training data
            y : array-like of shape (n_samples,)
                Target values

        Returns:
            self : returns an instance of self.
        """
        X, y = self.cook_training_set(y, X)

        if self.backend == "cpu":
            # Use numpy for CPU
            X_scaled, y_centered, self.X_mean_, self.y_mean_, self.X_scale_ = (
                self._center_scale_xy(np.array(X), np.array(y))
            )

            # SVD decomposition
            U, d, Vt = np.linalg.svd(X_scaled, full_matrices=False)

            # Compute coefficients
            rhs = U.T @ y_centered
            d2 = d**2

            if np.isscalar(self.lambda_):
                div = d2 + self.lambda_
                a = (d * rhs) / div
                self.coef_ = (Vt.T @ a) / self.X_scale_
            else:
                coefs = []
                for lambda_ in self.lambda_:
                    div = d2 + lambda_
                    a = (d * rhs) / div
                    coef = (Vt.T @ a) / self.X_scale_
                    coefs.append(coef)
                self.coef_ = np.array(coefs).T

        else:
            # Use JAX for GPU/TPU
            X_scaled, y_centered, self.X_mean_, self.y_mean_, self.X_scale_ = (
                self._center_scale_xy(jnp.array(X), jnp.array(y))
            )

            # SVD decomposition
            U, d, Vt = jnp.linalg.svd(X_scaled, full_matrices=False)

            # Compute coefficients
            rhs = mo.safe_sparse_dot(U.T, y_centered, backend=self.backend)
            d2 = d**2

            if np.isscalar(self.lambda_):
                div = d2 + self.lambda_
                a = (d * rhs) / div
                self.coef_ = (
                    mo.safe_sparse_dot(Vt.T, a, backend=self.backend) / self.X_scale_
                )
            else:
                coefs = []
                for lambda_ in self.lambda_:
                    div = d2 + lambda_
                    a = (d * rhs) / div
                    coef = (
                        mo.safe_sparse_dot(Vt.T, a, backend=self.backend)
                        / self.X_scale_
                    )
                    coefs.append(coef)
                self.coef_ = jnp.array(coefs).T

        # Compute GCV, HKB and LW criteria
        y_pred = self.predict(X)
        resid = y - y_pred
        n, p = X.shape
        s2 = jnp.sum(resid**2) / (n - p)

        if self.backend == "cpu":
            self.HKB_ = (p - 2) * s2 / jnp.sum(self.coef_**2)
            self.LW_ = (p - 2) * s2 * n / jnp.sum(y_pred**2)

            if np.isscalar(self.lambda_):
                div = d2 + self.lambda_
                self.GCV_ = jnp.sum((y - y_pred) ** 2) / (n - jnp.sum(d2 / div)) ** 2
            else:
                self.GCV_ = []
                for lambda_ in self.lambda_:
                    div = d2 + lambda_
                    gcv = jnp.sum((y - y_pred) ** 2) / (n - jnp.sum(d2 / div)) ** 2
                    self.GCV_.append(gcv)
                self.GCV_ = jnp.array(self.GCV_)
        else:
            self.HKB_ = (p - 2) * s2 / jnp.sum(self.coef_**2)
            self.LW_ = (p - 2) * s2 * n / jnp.sum(y_pred**2)

            if np.isscalar(self.lambda_):
                div = d2 + self.lambda_
                self.GCV_ = jnp.sum((y - y_pred) ** 2) / (n - jnp.sum(d2 / div)) ** 2
            else:
                gcvs = []
                for lambda_ in self.lambda_:
                    div = d2 + lambda_
                    gcv = jnp.sum((y - y_pred) ** 2) / (n - jnp.sum(d2 / div)) ** 2
                    gcvs.append(gcv)
                self.GCV_ = jnp.array(gcvs)

        return self

    def predict(self, X):
        """Predict using the Ridge regression model.

        Parameters:
            X : array-like of shape (n_samples, n_features)
                Samples to predict for

        Returns:
            y_pred : array-like of shape (n_samples,)
                Returns predicted values.
        """
        X = self.cook_test_set(X)

        if self.backend == "cpu":
            if np.isscalar(self.lambda_):
                return (
                    mo.safe_sparse_dot(X, self.coef_, backend=self.backend)
                    + self.y_mean_
                )
            else:
                return jnp.array(
                    [
                        mo.safe_sparse_dot(X, coef, backend=self.backend) + self.y_mean_
                        for coef in self.coef_.T
                    ]
                ).T
        else:
            if np.isscalar(self.lambda_):
                return (
                    mo.safe_sparse_dot(X, self.coef_, backend=self.backend)
                    + self.y_mean_
                )
            else:
                return jnp.array(
                    [
                        mo.safe_sparse_dot(X, coef, backend=self.backend) + self.y_mean_
                        for coef in self.coef_.T
                    ]
                ).T
