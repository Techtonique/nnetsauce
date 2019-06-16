"""RNN model"""

# Authors: Thierry Moudiki
#
# License: BSD 3

import numpy as np
from ..base import Base
from ..utils import matrixops as mo
from ..utils import misc as mx
from ..simulation import nodesimulation as ns
from ..simulation import rowsubsampling as rs
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
)



class RNN(Base):
    """RNN model class derived from class Base
    
       Parameters
       ----------
       obj: object
           any object containing a method fit (obj.fit()) and a method predict 
           (obj.predict())
       n_steps: int
           number of steps
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
        n_steps=3,
        n_hidden_features=5,
        activation_name="relu",
        a=0.01,
        nodes_sim="sobol",
        bias=True,
        dropout=0,
        direct_link=True, # 
        n_clusters=2,
        type_clust="kmeans",
        type_scaling=("std", "std", "std"),
        col_sample=1, # probably don't want to subsample here
        row_sample=1, # probably don't want to subsample here
        seed=123,
    ):

        super().__init__(
            n_hidden_features=n_hidden_features,
            activation_name=activation_name,
            a=a,
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

        self.obj = obj
        self.W_x = None
        self.W_h = None
        self.H_train = None # current state on training # To be defined when fit is called
        self.H_test = None # current state on test set # To be defined when predict is called
        self.n_steps = n_steps
        self.nodes_sim = nodes_sim

        
    def create_layer(self, scaled_X, W_x=None, W_h=None, training=True):
        """ Create hidden layer, updates state. """
        # training == True: to avoid updating the state twice        
        
        # only one predictor
        if len(scaled_X.shape) == 1:
            scaled_X = scaled_X.reshape(-1, 1)
        
        n_obs, n_features = scaled_X.shape        
        
        if training == True:
            if self.H_train is None:               
                self.H_train = np.zeros((n_obs, self.n_hidden_features))
        else:
            if self.H_test is None:
                self.H_test = np.zeros((n_obs, self.n_hidden_features))
            

        if (
            self.bias != True
        ):  # no bias term in the hidden layer

            if (W_x is None) | (W_h is None):
                
                if self.nodes_sim == "sobol":
                     
                    self.W_x = ns.generate_sobol2(
                        n_dims=n_features,
                        n_points=self.n_hidden_features,
                    )
                    self.W_h = ns.generate_sobol2(
                        n_dims=self.n_hidden_features,
                        n_points=self.n_hidden_features,
                    )

                if self.nodes_sim == "hammersley":

                    self.W_x = ns.generate_hammersley(
                        n_dims=n_features,
                        n_points=self.n_hidden_features,
                    )
                    self.W_h = ns.generate_hammersley(
                        n_dims=self.n_hidden_features,
                        n_points=self.n_hidden_features,
                    )

                if self.nodes_sim == "uniform":

                    self.W_x = ns.generate_uniform(
                        n_dims=n_features,
                        n_points=self.n_hidden_features,
                        seed=self.seed,
                    )
                    self.W_h = ns.generate_uniform(
                        n_dims=self.n_hidden_features,
                        n_points=self.n_hidden_features,
                        seed=self.seed,
                    )

                if self.nodes_sim == "halton":

                    self.W_x = ns.generate_halton(
                        n_dims=n_features,
                        n_points=self.n_hidden_features,
                    )
                    self.W_h = ns.generate_halton(
                        n_dims=self.n_hidden_features,
                        n_points=self.n_hidden_features,
                    )
                            
                assert (
                    scaled_X.shape[1] == self.W_x.shape[0]
                ), "check dimensions of covariates X and matrix W_x"
                
                assert (
                    self.H_train.shape[1] == self.W_h.shape[0]
                ), "check dimensions of state self.H and matrix W_h"
                
                if training == True:
                    self.H_train = mo.dropout(x=self.activation_func(
                            np.dot(scaled_X, self.W_x) + np.dot(self.H_train, self.W_h)),
                            drop_prob=self.dropout,
                            seed=self.seed)

                    return self.H_train
                
                else: 
                    self.H_test = mo.dropout(x=self.activation_func(
                            np.dot(scaled_X, self.W_x) + np.dot(self.H_test, self.W_h)),
                            drop_prob=self.dropout,
                            seed=self.seed)
                    
                    return self.H_test

            else:  # (W_x is not none) & (self.bias != True)
                
                assert (W_x is not None) & (W_h is not None), "W_x and W_h must be provided"

                assert (
                    scaled_X.shape[1] == W_x.shape[0]
                ), "check dimensions of covariates X and matrix W"

                # self.W = W
                if training == True:
                    self.H_train = mo.dropout(x=self.activation_func(
                            np.dot(scaled_X, W_x) + np.dot(self.H_train, W_h)),
                            drop_prob=self.dropout,
                            seed=self.seed)
                    
                    return self.H_train
                else: 
                    self.H_test = mo.dropout(x=self.activation_func(
                            np.dot(scaled_X, W_x) + np.dot(self.H_test, W_h)),
                            drop_prob=self.dropout,
                            seed=self.seed)                    
                    
                    return self.H_test

        else:  # self.bias == True

            if (W_x is None) | (W_h is None):

                n_features_1 = n_features + 1 # one more; constant feature

                if self.nodes_sim == "sobol":

                    self.W_x = ns.generate_sobol2(
                        n_dims=n_features_1,
                        n_points=self.n_hidden_features,
                    )
                    self.W_h = ns.generate_sobol2(
                        n_dims=self.n_hidden_features,
                        n_points=self.n_hidden_features,
                    )

                if self.nodes_sim == "hammersley":

                    self.W_x = ns.generate_hammersley(
                        n_dims=n_features_1,
                        n_points=self.n_hidden_features,
                    )
                    self.W_h = ns.generate_hammersley(
                        n_dims=self.n_hidden_features,
                        n_points=self.n_hidden_features,
                    )

                if self.nodes_sim == "uniform":

                    self.W_x = ns.generate_uniform(
                        n_dims=n_features_1,
                        n_points=self.n_hidden_features,
                        seed=self.seed,
                    )
                    self.W_h = ns.generate_uniform(
                        n_dims=self.n_hidden_features,
                        n_points=self.n_hidden_features,
                        seed=self.seed,
                    )
                    

                if self.nodes_sim == "halton":

                    self.W_x = ns.generate_halton(
                        n_dims=n_features_1,
                        n_points=self.n_hidden_features,
                    )
                    self.W_h = ns.generate_halton(
                        n_dims=self.n_hidden_features,
                        n_points=self.n_hidden_features,
                    )
                
                print(" in RNN::create_layer: ")
                print("scaled_X.shape")
                print(scaled_X.shape)
                print("self.W_x.shape")
                print(self.W_x.shape)
                print("self.H_train.shape")
                print(self.H_train.shape)
                print("self.W_h.shape")
                print(self.W_h.shape)
                
                if training == True:
                    self.H_train = mo.dropout(
                        x=self.activation_func(
                            np.dot(mo.cbind(np.ones(scaled_X.shape[0]), scaled_X),
                                   self.W_x) + np.dot(self.H_train, self.W_h)),
                        drop_prob=self.dropout,
                        seed=self.seed)
                        
                    return self.H_train
                else:
                    self.H_test = mo.dropout(
                        x=self.activation_func(
                            np.dot(mo.cbind(np.ones(scaled_X.shape[0]), scaled_X),
                                   self.W_x) + np.dot(self.H_test, self.W_h)),
                        drop_prob=self.dropout,
                        seed=self.seed)
                        
                    return self.H_test
                    

            else: # W_x is not None & self.bias == True
                
                assert (W_x is not None) & (W_h is not None), "W_x and W_h must be provided"
                
                # self.W = W
                if training == True:
                    self.H_train = mo.dropout(
                        x=self.activation_func(
                            np.dot(mo.cbind(
                                    np.ones(scaled_X.shape[0]),
                                    scaled_X), W_x) + np.dot(self.H_train, W_h)),
                        drop_prob=self.dropout,
                        seed=self.seed,
                    )
                    
                    return self.H_train
                else:
                    print("scaled_X.shape")
                    print(scaled_X.shape)
                    print("W_x.shape")
                    print(W_x.shape)
                    print("self.H_test.shape")
                    print(self.H_test.shape)
                    print("W_h.shape")
                    print(W_h.shape)
                    
                    self.H_test = mo.dropout(
                        x=self.activation_func(
                            np.dot(mo.cbind(
                                    np.ones(scaled_X.shape[0]),
                                    scaled_X), W_x) + np.dot(self.H_test, W_h)),
                        drop_prob=self.dropout,
                        seed=self.seed,
                    )
                    
                    return self.H_test
                    
        
        
     # redefined from Base, with 2 Ws: Wx and Wh
    def cook_training_set(
        self, y=None, X=None, W_x=None, W_h=None, **kwargs
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

                nn_scaler.fit(input_X)
                scaled_X = nn_scaler.transform(input_X)
                self.nn_scaler = nn_scaler

                # def create_layer(self, scaled_X, W_x=None, W_h=None):
                if (W_x is None) | (W_h is None):
                    Phi_X = self.create_layer(scaled_X, training=True)
                else:
                    Phi_X = self.create_layer(scaled_X, W_x=W_x, W_h=W_h, 
                                              training=True)

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

                if (W_x is None) | (W_h is None):
                    Phi_X = self.create_layer(scaled_X, training=True)
                else:
                    Phi_X = self.create_layer(scaled_X, W_x=W_x, W_h=W_h, 
                                              training=True)

                if self.direct_link == True:
                    print("augmented_X.shape")
                    print(augmented_X.shape)
                    print("Phi_X")
                    print(Phi_X)
                    Z = mo.cbind(augmented_X, Phi_X)
                else:
                    Z = Phi_X

                scaler.fit(Z)
                self.scaler = scaler

            else:  # no hidden layer

                Z = augmented_X
                scaler.fit(Z)
                self.scaler = scaler

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

            else:  # classification

                return (
                    y[self.index_row].reshape(n_row_sample),
                    self.scaler.transform(
                        Z[self.index_row, :].reshape(
                            n_row_sample, p
                        )
                    ),
                )

        else:  # y is not subsampled

            if mx.is_factor(y) == False:  # regression

                return (
                    centered_y,
                    self.scaler.transform(Z),
                )

            else:  # classification

                return (y, self.scaler.transform(Z))


    # redefined from Base, with 2 Ws: Wx and Wh
    def cook_test_set(self, X, **kwargs):
        """ Transform data from test set, with hidden layer. """                

        if (
            self.n_clusters <= 0
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

                # def create_layer(self, scaled_X, W_x=None, W_h=None):
                Phi_X = self.create_layer(scaled_X, W_x = self.W_x, 
                                          W_h = self.W_h, training = False)

                if self.direct_link == True:
                    return self.scaler.transform(
                        mo.cbind(X, Phi_X)
                    )
                else:
                    return self.scaler.transform(Phi_X)

            else:  # if no hidden layer

                return self.scaler.transform(X)

        else:  # data with clustering: self.n_clusters is None -----

            if self.col_sample == 1:

                predicted_clusters = self.encode_clusters(
                    X=X, predict=True, **kwargs
                )
                augmented_X = mo.cbind(
                    X, predicted_clusters
                )

            else:

                predicted_clusters = self.encode_clusters(
                    X=X[:, self.index_col],
                    predict=True,
                    **kwargs
                )
                augmented_X = mo.cbind(
                    X[:, self.index_col], predicted_clusters
                )

            if (
                self.n_hidden_features > 0
            ):  # if hidden layer

                scaled_X = self.nn_scaler.transform(
                    augmented_X
                )
                Phi_X = self.create_layer(scaled_X, W_x = self.W_x, 
                                          W_h = self.W_h, training = False)

                if self.direct_link == True:
                    return self.scaler.transform(
                        mo.cbind(augmented_X, Phi_X)
                    )
                else:
                    return self.scaler.transform(Phi_X)

            else:  # if no hidden layer

                return self.scaler.transform(augmented_X)
