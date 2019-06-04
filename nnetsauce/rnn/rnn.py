"""RNN model"""

# Authors: Thierry Moudiki
#
# License: BSD 3

import numpy as np
from ..base import Base
from ..utils import matrixops as mo
from ..simulation import nodesimulation as ns

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
       nodes_sim_x: str
           type of simulation for the nodes: 'sobol', 'hammersley', 'halton', 
           'uniform'
       nodes_sim_h: str
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
        nodes_sim_x="sobol",
        nodes_sim_h="uniform",
        bias=True,
        dropout=0,
        direct_link=True,
        n_clusters=2,
        type_clust="kmeans",
        type_scaling=("std", "std", "std"),
        col_sample=1,
        row_sample=1,
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
        self.H = None # state # To be defined when fit is called
        self.n_steps = n_steps
        self.nodes_sim_x=nodes_sim_x
        self.nodes_sim_h=nodes_sim_h


# in fit --> loop on n_steps
# rnn = RNN()        
# rnn.create_layer(X, self.H) # step 1 - self.H is identically 0 
# rnn.create_layer(X, self.H) # step 2 with updated self.H
# ... ...        

    def create_layer(self, scaled_X, W_x=None, W_h=None):
        """ Create hidden layer. """

        n_features = scaled_X.shape[1]

        if (
            self.bias != True
        ):  # no bias term in the hidden layer

            if W_x is None:
                
                assert self.W_h is not None, "self.W_h must be 'not None'"
                
                if self.nodes_sim_x == "sobol":
                    self.W_x = ns.generate_sobol2(
                        n_dims=n_features,
                        n_points=self.n_hidden_features,
                    )

                if self.nodes_sim_x == "hammersley":
                    self.W_x = ns.generate_hammersley(
                        n_dims=n_features,
                        n_points=self.n_hidden_features,
                    )

                if self.nodes_sim_x == "uniform":
                    self.W_x = ns.generate_uniform(
                        n_dims=n_features,
                        n_points=self.n_hidden_features,
                        seed=self.seed,
                    )

                if self.nodes_sim_x == "halton":
                    self.W_x = ns.generate_halton(
                        n_dims=n_features,
                        n_points=self.n_hidden_features,
                    )
                
                if self.nodes_sim_h == "sobol":
                    self.W_h = ns.generate_sobol2(
                        n_dims=n_features,
                        n_points=self.n_hidden_features,
                    )

                if self.nodes_sim_h == "hammersley":
                    self.W_h = ns.generate_hammersley(
                        n_dims=n_features,
                        n_points=self.n_hidden_features,
                    )

                if self.nodes_sim_h == "uniform":
                    self.W_h = ns.generate_uniform(
                        n_dims=n_features,
                        n_points=self.n_hidden_features,
                        seed=self.seed,
                    )

                if self.nodes_sim_h == "halton":
                    self.W_h = ns.generate_halton(
                        n_dims=n_features,
                        n_points=self.n_hidden_features,
                    )

                assert (
                    scaled_X.shape[1] == self.W.shape[0]
                ), "check dimensions of covariates X and matrix W"

                return mo.dropout(x=self.activation_func(
                        np.dot(scaled_X, self.W_x) + np.dot(self.H, self.W_h)),
                        drop_prob=self.dropout,
                        seed=self.seed)

            else:  # W_x is not none
                
                assert W_h is not None, "W_h must be provided"

                assert (
                    scaled_X.shape[1] == W_x.shape[0]
                ), "check dimensions of covariates X and matrix W"

                # self.W = W
                return mo.dropout(x=self.activation_func(
                        np.dot(scaled_X, W_x) + np.dot(self.H, W_h)),
                        drop_prob=self.dropout,
                        seed=self.seed)

        else:  # with bias term in the hidden layer

            if W_x is None:

                n_features_1 = n_features + 1

                if self.nodes_sim_x == "sobol":
                    self.W = ns.generate_sobol2(
                        n_dims=n_features_1,
                        n_points=self.n_hidden_features,
                    )

                if self.nodes_sim_x == "hammersley":
                    self.W = ns.generate_hammersley(
                        n_dims=n_features_1,
                        n_points=self.n_hidden_features,
                    )

                if self.nodes_sim_x == "uniform":
                    self.W = ns.generate_uniform(
                        n_dims=n_features_1,
                        n_points=self.n_hidden_features,
                        seed=self.seed,
                    )

                if self.nodes_sim_x == "halton":
                    self.W = ns.generate_halton(
                        n_dims=n_features_1,
                        n_points=self.n_hidden_features,
                    )
                
                if self.nodes_sim_h == "sobol":
                    self.W = ns.generate_sobol2(
                        n_dims=n_features_1,
                        n_points=self.n_hidden_features,
                    )

                if self.nodes_sim_h == "hammersley":
                    self.W = ns.generate_hammersley(
                        n_dims=n_features_1,
                        n_points=self.n_hidden_features,
                    )

                if self.nodes_sim_h == "uniform":
                    self.W = ns.generate_uniform(
                        n_dims=n_features_1,
                        n_points=self.n_hidden_features,
                        seed=self.seed,
                    )

                if self.nodes_sim_h == "halton":
                    self.W = ns.generate_halton(
                        n_dims=n_features_1,
                        n_points=self.n_hidden_features,
                    )


                return mo.dropout(
                    x=self.activation_func(
                        np.dot(mo.cbind(np.ones(scaled_X.shape[0]), scaled_X),
                               self.W_x) + np.dot(mo.cbind(np.ones(self.H.shape[0]),
                                self.H), self.W_h)),
                    drop_prob=self.dropout,
                    seed=self.seed,
                )

            else: # W_x is not None 

                # self.W = W
                return mo.dropout(
                    x=self.activation_func(
                        np.dot(mo.cbind(
                                np.ones(scaled_X.shape[0]),
                                scaled_X), W_x) + np.dot(
                            mo.cbind(
                                np.ones(self.H.shape[0]),
                                self.H,
                            ),W_h)),
                    drop_prob=self.dropout,
                    seed=self.seed,
                )