import numpy as np
from numpy import linalg as la
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from ..utils import matrixops as mo
from ..simulation import nodesimulation as ns

# do DOCSTRINGS
# do GCV
class Base(object):
    
    
    # construct the object -----
    
    def __init__(self, n_hidden_features=5, 
                 activation_name='relu',
                 nodes_sim='sobol',
                 n_clusters=None,
                 seed = 123):
        """Base RVFL model with no regularization
        
           Parameters
           ----------
           n_hidden_features: int
               number of nodes in the hidden layer
           activation_name: str
               activation function: 'relu', 'tanh' or 'sigmoid'
        """
        
        activation_options = {
            'relu': lambda x: np.maximum(x, 0),
            'tanh': lambda x: np.tanh(x),
            'sigmoid': lambda x: 1/(1+np.exp(-x))}
        
        self.n_hidden_features = n_hidden_features
        self.activation_name = activation_name
        self.activation_func = activation_options[activation_name]
        self.nodes_sim = nodes_sim
        self.seed = seed
        self.n_clusters = n_clusters
        self.kmeans = None
        self.kmeans_scaler = None
        self.nn_scaler = None
        self.scaler = None
        self.encoder = None
        self.W = None
        self.X = None
        self.y = None
        self.y_mean = None
        self.beta = None

    
    # getter -----
    
    def get_params(self):
        
        return {'n_hidden_features': self.n_hidden_features, 
                'activation_name': self.activation_name, 
                'activation_func': self.activation_func,
                'nodes_sim': self.nodes_sim,
                'seed': self.seed,
                'n_clusters': self.n_clusters,
                'nn_scaler': self.nn_scaler,  
                'scaler': self.scaler,  
                'W': self.W, 
                'y_mean': self.y_mean,                
                'beta': self.beta}
    
    
    # setter -----
    
    def set_params(self, n_hidden_features=5, 
                   activation_name='relu', 
                   nodes_sim='sobol',
                   n_clusters=None,
                   seed=123):
        
        activation_options = {
            'relu': lambda x: np.maximum(x, 0),
            'tanh': lambda x: np.tanh(x),
            'sigmoid': lambda x: 1/(1+np.exp(-x))}
        
        self.n_hidden_features = n_hidden_features
        self.activation_name = activation_name
        self.activation_func = activation_options[activation_name]
        self.nodes_sim = nodes_sim
        self.n_clusters = n_clusters
        self.seed = seed
    
    
    # fit -----
    
    def fit(self, X, y):

        centered_y, scaled_Z = self.preproc_training_set(y = y, X = X)
        self.beta = la.lstsq(scaled_Z, centered_y)[0]
        
        return self            
        
    
    # predict -----
    
    def predict(self, X):
    
        # convert X to np.array with 
        # 2 dimensions if it's a vector 
        if len(X.shape) == 1:
            X = mo.to_np_array(X)
            
        return self.y_mean + np.dot(self.preproc_test_set(X), 
                                        self.beta)

    # preprocessing methods to be inherited -----
    
    
    # create new covariates with kmeans clustering
    def encode_kmeans(self, X=None, n_clusters=2, 
                      predict=False, **kwargs):
        
        if (X is None):
            input_X = self.X
        else:
            input_X = X
                
        if predict == False: # encode training set 
            
            # scale input 
            scaler = StandardScaler(copy=True, 
                                    with_mean=True, 
                                    with_std=True)  
            scaler.fit(input_X)        
            scaled_X = scaler.transform(input_X)
            self.kmeans_scaler = scaler
    
            # do kmeans + one-hot encoding
            kmeans = KMeans(n_clusters=n_clusters)
            kmeans.fit(scaled_X)
            X_kmeans = kmeans.predict(scaled_X)
            self.kmeans = kmeans
            self.encoder = OneHotEncoder(**kwargs)
    
            return self.encoder.fit_transform(X_kmeans.reshape(-1, 1)).toarray()
            
        else: # if predict == True, encode test set
            
            X_kmeans = self.kmeans.predict(self.kmeans_scaler.transform(X))
            return self.encoder.fit_transform(X_kmeans.reshape(-1, 1)).toarray()
            
        
    # create hidden layer
    def create_layer(self, scaled_X, n_features, W=None):        
            
        if (W is None):
            if self.nodes_sim == 'sobol':
                self.W = ns.generate_sobol(n_dims=n_features, 
                                       n_points=self.n_hidden_features)
            
            if self.nodes_sim == 'hammersley':
                self.W = ns.generate_hammersley(n_dims=n_features, 
                                       n_points=self.n_hidden_features)
                
            if self.nodes_sim == 'uniform':
                self.W = ns.generate_uniform(n_dims=n_features, 
                                          n_points=self.n_hidden_features, 
                                          seed = self.seed)
            
            if self.nodes_sim == 'halton':
                self.W = ns.generate_halton(n_dims=n_features, 
                                         n_points=self.n_hidden_features)
            
            return self.activation_func(np.dot(scaled_X, self.W))
        else:
            self.W = W
            return self.activation_func(np.dot(scaled_X, W))
        
        
    # create new data for training set, with hidden layer, center the response    
    def preproc_training_set(self, y=None, X=None, W=None): 
        
        # either X and y are stored or not 
        #assert ((y is None) & (X is None)) | ((y is not None) & (X is not None))
        nn_scaler = StandardScaler(copy=True, 
                                    with_mean=True, 
                                    with_std=True)
            
        scaler = StandardScaler(copy=True, 
                                    with_mean=True, 
                                    with_std=True)  
            
        # center y
        if (y is None):
            y_mean = self.y.mean()
            centered_y = self.y - y_mean
        else:
            y_mean = y.mean()
            self.y_mean = y_mean
            centered_y = y - y_mean
            
        if (X is None):
            input_X = self.X
        else:
            input_X = X
        
        # 1 - data without clustering: self.n_clusters is None -----      
        if self.n_clusters is None: 
            
            n_features = input_X.shape[1]
            nn_scaler.fit(input_X)
            scaled_X = nn_scaler.transform(input_X)
            self.nn_scaler = nn_scaler
            
            # construct the hidden layer
#            if (W is None):
#                if self.nodes_sim == 'sobol':
#                    self.W = ns.generate_sobol(n_dims=n_features, 
#                                           n_points=self.n_hidden_features)
#                
#                if self.nodes_sim == 'hammersley':
#                    self.W = ns.generate_hammersley(n_dims=n_features, 
#                                           n_points=self.n_hidden_features)
#                    
#                if self.nodes_sim == 'uniform':
#                    self.W = ns.generate_uniform(n_dims=n_features, 
#                                              n_points=self.n_hidden_features, 
#                                              seed = self.seed)
#                
#                if self.nodes_sim == 'halton':
#                    self.W = ns.generate_halton(n_dims=n_features, 
#                                             n_points=self.n_hidden_features)
#                
#                Phi_X = self.activation_func(np.dot(scaled_X, self.W))
#            else:
#                self.W = W
#                Phi_X = self.activation_func(np.dot(scaled_X, W))
            
            if (W is None):
                Phi_X = self.create_layer(scaled_X, n_features)
            else:
                Phi_X = self.create_layer(scaled_X, n_features, W=W)
            
            Z = mo.cbind(input_X, Phi_X)
            scaler.fit(Z)
            self.scaler = scaler 
        
        # 2 - data with clustering: self.n_clusters is not None -----  
        else: 
            
            n_features = input_X.shape[1] + self.n_clusters 
            augmented_X = mo.cbind(input_X, self.encode_kmeans(X = input_X, 
                                                  n_clusters = self.n_clusters))
            nn_scaler.fit(augmented_X)
            scaled_X = nn_scaler.transform(augmented_X)           
            self.nn_scaler = nn_scaler
            
            # construct the hidden layer
#            if (W is None):
#                if self.nodes_sim == 'sobol':
#                    self.W = ns.generate_sobol(n_dims=n_features, 
#                                           n_points=self.n_hidden_features)
#                
#                if self.nodes_sim == 'hammersley':
#                    self.W = ns.generate_hammersley(n_dims=n_features, 
#                                           n_points=self.n_hidden_features)
#                    
#                if self.nodes_sim == 'uniform':
#                    self.W = ns.generate_uniform(n_dims=n_features, 
#                                              n_points=self.n_hidden_features, 
#                                              seed = self.seed)
#                
#                if self.nodes_sim == 'halton':
#                    self.W = ns.generate_halton(n_dims=n_features, 
#                                             n_points=self.n_hidden_features)
#                
#                Phi_X = self.activation_func(np.dot(scaled_X, self.W))
#            else:
#                self.W = W
#                Phi_X = self.activation_func(np.dot(scaled_X, W))
            
            if (W is None):
                Phi_X = self.create_layer(scaled_X, n_features)
            else:
                Phi_X = self.create_layer(scaled_X, n_features, W=W)
            
            Z = mo.cbind(augmented_X, Phi_X)
            scaler.fit(Z)
            self.scaler = scaler 

        return centered_y, self.scaler.transform(Z) 
    
    
    # transform data from test set, with hidden layers
    def preproc_test_set(self, X):
        
        # convert X to np.array with 
        # 2 dimensions if it's a vector 
        if len(X.shape) == 1:
            X = mo.to_np_array(X)
        
        # 1 - data without clustering: self.n_clusters is None -----      
        if self.n_clusters is None: 
            
            scaled_X = self.nn_scaler.transform(X)
            Phi_X = self.activation_func(np.dot(scaled_X, self.W))
           
            return self.scaler.transform(mo.cbind(X, Phi_X))
            
        # 2 - data with clustering: self.n_clusters is None -----      
        else:
            
            augmented_X = mo.cbind(X, self.encode_kmeans(X = X, 
                                                  n_clusters = self.n_clusters, 
                                                  predict = True))
            scaled_X = self.nn_scaler.transform(augmented_X)
            Phi_X = self.activation_func(np.dot(scaled_X, self.W))
            
            return self.scaler.transform(mo.cbind(augmented_X, Phi_X))
            
    
#if __name__ == '__main__':
#
#    from sklearn import datasets   
#    
#    # Example 1 -----
#    
#    n_features = 4
#    n_samples = 10
#    X, y = datasets.make_regression(n_features=n_features, 
#                       n_samples=n_samples, 
#                       random_state=0)
#
#    fit_obj = Base(n_hidden_features=3, 
#                 activation_name='relu', 
#                 n_clusters=2)
#    
#    centered_y, scaled_Z = fit_obj.preproc_training_set(y=y, X=X)
#
#    print(centered_y.shape)
#    print(scaled_Z.shape)
#    print(centered_y.mean())
#    print(scaled_Z.mean(axis = 0))
#    print(np.sqrt(scaled_Z.var(axis = 0)))
#    
#    fit_obj.fit(X, y) 
#    print(fit_obj.beta)
#    print(len(fit_obj.beta))
#    print(fit_obj.predict(X))
    
 
#    # Example 2 -----
    
#    diabetes = datasets.load_diabetes()
#    
#    # data snippet
#    diabetes.feature_names
#    
#    # shape 
#    diabetes.data.shape
#    diabetes.target.shape
#    
#    # define X and y
#    X = diabetes.data 
#    y = diabetes.target
#    
#    fit_obj = rvflBase(n_hidden_features=3, 
#                       activation_name='relu', 
#                       n_clusters=2)
#    
#    centered_y, scaled_Z = fit_obj.preproc_training_set(y=y, X=X)
#
#    print(centered_y.shape)
#    print(scaled_Z.shape)
#    print(centered_y.mean())
#    print(scaled_Z.mean(axis = 0))
#    print(np.sqrt(scaled_Z.var(axis = 0)))
#    
#    fit_obj.fit(X, y)    
#    z = fit_obj.predict(X) - y
#    print(fit_obj.predict(X))
#    print(y)
#    print(z.mean())
#
