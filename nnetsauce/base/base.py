"""Random Vector Functional Link Network regression."""

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
from ..utils import matrixops as mo
from ..utils import misc as mx
from ..simulation import nodesimulation as ns


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
       type_clust: str
           type of clustering method: currently k-means ('kmeans') or Gaussian 
           Mixture Model ('gmm')
       type_scaling: a tuple of 3 strings
           scaling methods for inputs, hidden layer, and clustering respectively
           (and when relevant). 
           Currently available: standardization ('std') or MinMax scaling ('minmax')
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
        type_clust="kmeans",
        type_scaling=("std", "std", "std"),
        seed=123,
    ):

        # input checks -----

        assert activation_name in (
            "relu",
            "tanh",
            "sigmoid",
            "prelu",
            "elu",
        ), "'activation_name' should be in ('relu', 'tanh', 'sigmoid','prelu', 'elu')"

        assert nodes_sim in (
            "sobol",
            "hammersley",
            "uniform",
            "halton",
        ), "'nodes_sim' should be in ('sobol', 'hammersley', 'uniform', 'halton')"

        assert type_clust in (
            "kmeans",
            "gmm",
        ), "'type_clust' should be in ('kmeans', 'gmm')"

        assert (len(type_scaling) == 3) & all(
            type_scaling[i] in ("minmax", "std")
            for i in range(len(type_scaling))
        ), "'type_scaling' must have length 3, and available scaling methods are 'minmax' scaling and standardization ('std')"

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
        self.type_clust = type_clust
        self.type_scaling = type_scaling
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

            scaling_options = {
                "std": StandardScaler(
                    copy=True, with_mean=True, with_std=True
                ),
                "minmax": MinMaxScaler(),
            }

            # scale input data before clustering
            scaler = scaling_options[self.type_scaling[2]]

            scaler.fit(X)
            scaled_X = scaler.transform(X)
            self.clustering_scaler = scaler

            if self.type_clust == "kmeans":

                # do kmeans + one-hot encoding
                np.random.seed(self.seed)
                kmeans = KMeans(
                    n_clusters=self.n_clusters, **kwargs
                )
                kmeans.fit(scaled_X)
                X_kmeans = kmeans.predict(scaled_X)
                self.clustering_obj = kmeans

                return mo.one_hot_encode(
                    X_kmeans, self.n_clusters
                )

            if self.type_clust == "gmm":

                np.random.seed(self.seed)
                gmm = GaussianMixture(
                    n_components=self.n_clusters, **kwargs
                )
                gmm.fit(scaled_X)
                X_gmm = gmm.predict(scaled_X)
                self.clustering_obj = gmm

                return mo.one_hot_encode(
                    X_gmm, self.n_clusters
                )

        else:  # if predict == True, encode test set

            X_clustered = self.clustering_obj.predict(
                self.clustering_scaler.transform(X)
            )

            return mo.one_hot_encode(
                X_clustered, self.n_clusters
            )


    def create_layer(self, scaled_X, W=None):
        """ Create hidden layer. """

        n_features = scaled_X.shape[1]

        if (
            self.bias != True
        ):  # no bias term in the hidden layer

            if W is None:

                if self.nodes_sim == "sobol":
                    self.W = ns.generate_sobol(
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
                    self.W = ns.generate_halton(
                        n_dims=n_features,
                        n_points=self.n_hidden_features,
                    )

                return mo.dropout(
                    x=self.activation_func(
                        np.dot(scaled_X, self.W)
                    ),
                    drop_prob=self.dropout,
                    seed=self.seed,
                )

            else:

                # self.W = W
                return mo.dropout(
                    x=self.activation_func(
                        np.dot(scaled_X, W)
                    ),
                    drop_prob=self.dropout,
                    seed=self.seed,
                )

        else:  # with bias term in the hidden layer

            if W is None:

                n_features_1 = n_features + 1

                if self.nodes_sim == "sobol":
                    self.W = ns.generate_sobol(
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

            else:

                # self.W = W
                return mo.dropout(
                    x=self.activation_func(
                        np.dot(
                            mo.cbind(
                                np.ones(scaled_X.shape[0]),
                                scaled_X,
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
        scaling_options = {
            "std": StandardScaler(
                copy=True, with_mean=True, with_std=True
            ),
            "minmax": MinMaxScaler(),
        }

        if self.n_hidden_features > 0:  # has a hidden layer

            assert len(self.type_scaling) >= 2, ""

            nn_scaling_options = {
                "std": StandardScaler(
                    copy=True, with_mean=True, with_std=True
                ),
                "minmax": MinMaxScaler(),
            }

            nn_scaler = nn_scaling_options[
                self.type_scaling[1]
            ]

        scaler = scaling_options[self.type_scaling[0]]

        # center y
        if mx.is_factor(y) == False:  # regression
            if y is None:
                y_mean = self.y.mean()
                centered_y = self.y - y_mean
            else:
                y_mean = y.mean()
                self.y_mean = y_mean
                centered_y = y - y_mean

        if X is None:
            input_X = self.X
        else:
            input_X = X

        if (
            self.n_clusters <= 0
        ):  # data without any clustering: self.n_clusters is None -----

            if (
                self.n_hidden_features > 0
            ):  # with hidden layer

                nn_scaler.fit(input_X)
                scaled_X = nn_scaler.transform(input_X)
                self.nn_scaler = nn_scaler

                if W is None:
                    Phi_X = self.create_layer(scaled_X)
                else:
                    Phi_X = self.create_layer(scaled_X, W=W)

                if self.direct_link == True:
                    Z = mo.cbind(input_X, Phi_X)
                else:
                    Z = Phi_X

                scaler.fit(Z)
                self.scaler = scaler

            else:  # no hidden layer

                Z = input_X
                scaler.fit(Z)
                self.scaler = scaler

        else:  # data with clustering: self.n_clusters is not None -----

            augmented_X = mo.cbind(
                input_X,
                self.encode_clusters(input_X, **kwargs),
            )

            if (
                self.n_hidden_features > 0
            ):  # with hidden layer

                nn_scaler.fit(augmented_X)
                scaled_X = nn_scaler.transform(augmented_X)
                self.nn_scaler = nn_scaler

                if W is None:
                    Phi_X = self.create_layer(scaled_X)
                else:
                    Phi_X = self.create_layer(scaled_X, W=W)

                if self.direct_link == True:
                    Z = mo.cbind(augmented_X, Phi_X)
                else:
                    Z = Phi_X

                scaler.fit(Z)
                self.scaler = scaler

            else:  # no hidden layer

                Z = augmented_X
                scaler.fit(Z)
                self.scaler = scaler

        if mx.is_factor(y) == False:  # regression

            return centered_y, self.scaler.transform(Z)

        else:  # classification

            return self.scaler.transform(Z)
        

    def cook_test_set(self, X, **kwargs):
        """ Transform data from test set, with hidden layer. """

        if (
            self.n_clusters <= 0
        ):  # data without clustering: self.n_clusters is None -----

            if (
                self.n_hidden_features > 0
            ):  # if hidden layer

                scaled_X = self.nn_scaler.transform(X)
                Phi_X = self.create_layer(scaled_X, self.W)

                if self.direct_link == True:
                    return self.scaler.transform(
                        mo.cbind(X, Phi_X)
                    )
                else:
                    return self.scaler.transform(Phi_X)

            else:  # if no hidden layer

                return self.scaler.transform(X)

        else:  # data with clustering: self.n_clusters is None -----

            predicted_clusters = self.encode_clusters(
                X=X, predict=True, **kwargs
            )
            augmented_X = mo.cbind(X, predicted_clusters)

            if (
                self.n_hidden_features > 0
            ):  # if hidden layer

                scaled_X = self.nn_scaler.transform(
                    augmented_X
                )
                Phi_X = self.create_layer(scaled_X, self.W)

                if self.direct_link == True:
                    return self.scaler.transform(
                        mo.cbind(augmented_X, Phi_X)
                    )
                else:
                    return self.scaler.transform(Phi_X)

            else:  # if no hidden layer

                return self.scaler.transform(augmented_X)
