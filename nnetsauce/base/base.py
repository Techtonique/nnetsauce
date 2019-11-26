"""Base class."""

# Authors: Thierry Moudiki
#
# License: BSD 3

import numpy as np
from functools import partial
from sklearn.base import BaseEstimator
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
)
from ..utils import activations as ac
from ..utils import memoize
from ..utils import matrixops as mo
from ..utils import misc as mx
from ..simulation import nodesimulation as ns
from ..simulation import rowsubsampling as rs


class Base(BaseEstimator):
    """Base model with direct link and nonlinear activation.
        
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
           indicates if the hidden layer contains a bias term (True) or 
           not (False)
       dropout: float
           regularization parameter; (random) percentage of nodes dropped out 
           of the training
       direct_link: boolean
           indicates if the original predictors are included (True) in model's 
           fitting or not (False)
       n_clusters: int
           number of clusters for type_clust='kmeans' or type_clust='gmm' 
           clustering (could be 0: no clustering)
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
           reproducibility seed for nodes_sim=='uniform', clustering and dropout
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
    ):

        # input checks -----

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
            type_scaling[i] in ("minmax", "std")
            for i in range(len(type_scaling))
        ), "'type_scaling' must have length 3, and available scaling methods are 'minmax' scaling and standardization ('std')"

        assert (col_sample >= 0) & (
            col_sample <= 1
        ), "'col_sample' must be comprised between 0 and 1 (both included)"

        # activation function -----

        activation_options = {
            "relu": ac.relu,
            "tanh": np.tanh,
            "sigmoid": ac.sigmoid,
            "prelu": partial(ac.prelu, a=a),
            "elu": partial(ac.elu, a=a),
        }

        self.n_hidden_features = n_hidden_features
        self.activation_name = activation_name
        self.a = a
        self.activation_func = activation_options[
            activation_name
        ]
        self.nodes_sim = nodes_sim
        self.bias = bias
        self.seed = seed
        self.dropout = dropout
        self.direct_link = direct_link
        self.cluster_encode = cluster_encode
        self.type_clust = type_clust
        self.type_scaling = type_scaling
        self.col_sample = col_sample
        self.row_sample = row_sample
        self.index_col = None
        self.index_row = True
        self.n_clusters = n_clusters
        self.clustering_obj = None
        self.clustering_scaler = None
        self.nn_scaler = None
        self.scaler = None
        self.encoder = None
        self.W = None
        self.X = None
        self.y = None
        self.y_mean = None
        self.beta = None

    # "preprocessing" methods to be inherited -----

    def encode_clusters(
        self, X=None, predict=False, **kwargs
    ):  #
        """ Create new covariates with kmeans or GMM clustering. 

        Parameters
        ----------
        X: {array-like}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number 
            of samples and n_features is the number of features.
        
        predict: boolean
            is False on training set and True on test set
        
        **kwargs: 
            additional parameters to be passed to the 
            clustering method  
            
        Returns
        -------
        clusters' matrix, one-hot encoded: {array-like}        
        """

        if X is None:
            X = self.X

        if predict == False:  # encode training set

            # scale input data before clustering
            self.clustering_scaler, scaled_X = mo.scale_covariates(
                X, choice=self.type_scaling[2]
            )

            if self.type_clust == "kmeans":

                self.clustering_obj, X_kmeans = mo.cluster_covariates(
                    scaled_X,
                    self.n_clusters,
                    self.seed,
                    type_clust="kmeans",
                    **kwargs
                )

                if self.cluster_encode == True:

                    return mo.one_hot_encode(
                        X_kmeans, self.n_clusters
                    )

                return X_kmeans

            if self.type_clust == "gmm":

                self.clustering_obj, X_gmm = mo.cluster_covariates(
                    scaled_X,
                    self.n_clusters,
                    self.seed,
                    type_clust="gmm",
                    **kwargs
                )

                if self.cluster_encode == True:

                    return mo.one_hot_encode(
                        X_gmm, self.n_clusters
                    )

                return X_gmm

        # if predict == True, encode test set
        X_clustered = self.clustering_obj.predict(
            self.clustering_scaler.transform(X)
        )

        if self.cluster_encode == True:

            return mo.one_hot_encode(
                X_clustered, self.n_clusters
            )

        return X_clustered

    def create_layer(self, scaled_X, W=None):
        """ Create hidden layer. """

        n_features = scaled_X.shape[1]

        if (
            self.bias is False
        ):  # no bias term in the hidden layer

            if W is None:

                if self.nodes_sim == "sobol":
                    try:  # try cpp version
                        self.W = ns.generate_sobol_cpp(
                            n_dims=n_features,
                            n_points=self.n_hidden_features,
                        )
                    except:  # python version
                        self.W = ns.generate_sobol2(
                            n_dims=n_features,
                            n_points=self.n_hidden_features,
                        )

                if self.nodes_sim == "hammersley":
                    self.W = ns.generate_hammersley(
                        n_dims=n_features,
                        n_points=self.n_hidden_features,
                    )

                if self.nodes_sim == "uniform":
                    self.W = ns.generate_uniform(
                        n_dims=n_features,
                        n_points=self.n_hidden_features,
                        seed=self.seed,
                    )

                if self.nodes_sim == "halton":
                    try:  # try cpp version
                        self.W = ns.generate_halton_cpp(
                            n_dims=n_features,
                            n_points=self.n_hidden_features,
                        )
                    except:  # python version
                        self.W = ns.generate_halton(
                            n_dims=n_features,
                            n_points=self.n_hidden_features,
                        )

                assert (
                    scaled_X.shape[1] == self.W.shape[0]
                ), "check dimensions of covariates X and matrix W"

                return mo.dropout(
                    x=self.activation_func(
                        np.dot(scaled_X, self.W)
                    ),
                    drop_prob=self.dropout,
                    seed=self.seed,
                )

            # W is not none
            assert (
                scaled_X.shape[1] == W.shape[0]
            ), "check dimensions of covariates X and matrix W"

            # self.W = W
            return mo.dropout(
                x=self.activation_func(np.dot(scaled_X, W)),
                drop_prob=self.dropout,
                seed=self.seed,
            )

        # with bias term in the hidden layer
        if W is None:

            n_features_1 = n_features + 1

            if self.nodes_sim == "sobol":
                try:  # try cpp version
                    self.W = ns.generate_sobol_cpp(
                        n_dims=n_features_1,
                        n_points=self.n_hidden_features,
                    )
                except:  # python version
                    self.W = ns.generate_sobol2(
                        n_dims=n_features_1,
                        n_points=self.n_hidden_features,
                    )

            if self.nodes_sim == "hammersley":
                self.W = ns.generate_hammersley(
                    n_dims=n_features_1,
                    n_points=self.n_hidden_features,
                )

            if self.nodes_sim == "uniform":
                self.W = ns.generate_uniform(
                    n_dims=n_features_1,
                    n_points=self.n_hidden_features,
                    seed=self.seed,
                )

            if self.nodes_sim == "halton":
                try:  # try cpp version
                    self.W = ns.generate_halton_cpp(
                        n_dims=n_features_1,
                        n_points=self.n_hidden_features,
                    )
                except:  # python version
                    self.W = ns.generate_halton(
                        n_dims=n_features_1,
                        n_points=self.n_hidden_features,
                    )

            return mo.dropout(
                x=self.activation_func(
                    np.dot(
                        mo.cbind(
                            np.ones(scaled_X.shape[0]),
                            scaled_X,
                        ),
                        self.W,
                    )
                ),
                drop_prob=self.dropout,
                seed=self.seed,
            )

        # W is not None
        # self.W = W
        return mo.dropout(
            x=self.activation_func(
                np.dot(
                    mo.cbind(
                        np.ones(scaled_X.shape[0]), scaled_X
                    ),
                    W,
                )
            ),
            drop_prob=self.dropout,
            seed=self.seed,
        )

    def cook_training_set(
        self, y=None, X=None, W=None, **kwargs
    ):
        """ Create new data for training set, with hidden layer, center the response. """

        # either X and y are stored or not
        # assert ((y is None) & (X is None)) | ((y is not None) & (X is not None))
        if self.n_hidden_features > 0:  # has a hidden layer

            assert len(self.type_scaling) >= 2, ""

        # center y
        if mx.is_factor(y) == False:  # regression

            if y is None:
                self.y_mean, centered_y = mo.center_response(
                    self.y
                )
            else:  # keep
                self.y_mean, centered_y = mo.center_response(
                    y
                )

        if X is None:

            if self.col_sample == 1:

                input_X = self.X

            else:

                n_features = self.X.shape[1]
                new_n_features = int(
                    np.ceil(n_features * self.col_sample)
                )
                assert (
                    new_n_features >= 1
                ), "check class attribute 'col_sample' and the number of covariates provided for X"
                np.random.seed(self.seed)
                index_col = np.random.choice(
                    range(n_features),
                    size=new_n_features,
                    replace=False,
                )
                self.index_col = index_col
                input_X = self.X[:, self.index_col]

        else:  # X is not None

            if self.col_sample == 1:

                input_X = X

            else:

                n_features = X.shape[1]
                new_n_features = int(
                    np.ceil(n_features * self.col_sample)
                )
                assert (
                    new_n_features >= 1
                ), "check class attribute 'col_sample' and the number of covariates provided for X"
                np.random.seed(self.seed)
                index_col = np.random.choice(
                    range(n_features),
                    size=new_n_features,
                    replace=False,
                )
                self.index_col = index_col
                input_X = X[:, self.index_col]

        if (
            self.n_clusters <= 0
        ):  # data without any clustering: self.n_clusters is None -----

            if (
                self.n_hidden_features > 0
            ):  # with hidden layer

                self.nn_scaler, scaled_X = mo.scale_covariates(
                    input_X, choice=self.type_scaling[1]
                )

                if W is None:
                    Phi_X = self.create_layer(scaled_X)
                else:
                    Phi_X = self.create_layer(scaled_X, W=W)

                if self.direct_link == True:
                    Z = mo.cbind(input_X, Phi_X)
                else:
                    Z = Phi_X

                self.scaler, scaled_Z = mo.scale_covariates(
                    Z, choice=self.type_scaling[0]
                )

            else:  # no hidden layer

                Z = input_X

                self.scaler, scaled_Z = mo.scale_covariates(
                    Z, choice=self.type_scaling[0]
                )

        else:  # data with clustering: self.n_clusters is not None -----

            augmented_X = mo.cbind(
                input_X,
                self.encode_clusters(input_X, **kwargs),
            )

            if (
                self.n_hidden_features > 0
            ):  # with hidden layer

                self.nn_scaler, scaled_X = mo.scale_covariates(
                    augmented_X, choice=self.type_scaling[1]
                )

                if W is None:
                    Phi_X = self.create_layer(scaled_X)
                else:
                    Phi_X = self.create_layer(scaled_X, W=W)

                if self.direct_link == True:
                    Z = mo.cbind(augmented_X, Phi_X)
                else:
                    Z = Phi_X

                self.scaler, scaled_Z = mo.scale_covariates(
                    Z, choice=self.type_scaling[0]
                )

            else:  # no hidden layer

                Z = augmented_X

                self.scaler, scaled_Z = mo.scale_covariates(
                    Z, choice=self.type_scaling[0]
                )

        # Returning model inputs -----

        # y is subsampled
        if self.row_sample < 1:

            n, p = Z.shape

            if y is None:

                self.index_row = rs.subsample(
                    y=self.y,
                    row_sample=self.row_sample,
                    seed=self.seed,
                )

            else:

                self.index_row = rs.subsample(
                    y=y,
                    row_sample=self.row_sample,
                    seed=self.seed,
                )

            n_row_sample = len(self.index_row)

            if mx.is_factor(y) == False:  # regression

                return (
                    centered_y[self.index_row].reshape(
                        n_row_sample
                    ),
                    self.scaler.transform(
                        Z[self.index_row, :].reshape(
                            n_row_sample, p
                        )
                    ),
                )

            # classification
            return (
                y[self.index_row].reshape(n_row_sample),
                self.scaler.transform(
                    Z[self.index_row, :].reshape(
                        n_row_sample, p
                    )
                ),
            )

        # y is not subsampled
        if mx.is_factor(y) == False:  # regression

            return (centered_y, self.scaler.transform(Z))

        # classification
        return (y, self.scaler.transform(Z))

    def cook_test_set(self, X, **kwargs):
        """ Transform data from test set, with hidden layer. """

        if (
            self.n_clusters == 0
        ):  # data without clustering: self.n_clusters is None -----

            if (
                self.n_hidden_features > 0
            ):  # if hidden layer

                if self.col_sample == 1:

                    scaled_X = self.nn_scaler.transform(X)

                else:

                    scaled_X = self.nn_scaler.transform(
                        X[:, self.index_col]
                    )

                Phi_X = self.create_layer(scaled_X, self.W)

                if self.direct_link == True:

                    return self.scaler.transform(
                        mo.cbind(scaled_X, Phi_X)
                    )

                # when self.direct_link == False
                return self.scaler.transform(Phi_X)

            # if no hidden layer # self.n_hidden_features == 0
            return self.scaler.transform(X)

        # data with clustering: self.n_clusters > 0 -----
        if self.col_sample == 1:

            predicted_clusters = self.encode_clusters(
                X=X, predict=True, **kwargs
            )
            augmented_X = mo.cbind(X, predicted_clusters)

        else:

            predicted_clusters = self.encode_clusters(
                X=X[:, self.index_col],
                predict=True,
                **kwargs
            )
            augmented_X = mo.cbind(
                X[:, self.index_col], predicted_clusters
            )

        if self.n_hidden_features > 0:  # if hidden layer

            scaled_X = self.nn_scaler.transform(augmented_X)
            Phi_X = self.create_layer(scaled_X, self.W)

            if self.direct_link == True:
                return self.scaler.transform(
                    mo.cbind(augmented_X, Phi_X)
                )

            return self.scaler.transform(Phi_X)

        # if no hidden layer
        return self.scaler.transform(augmented_X)
