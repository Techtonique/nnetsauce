from ..base import Base


class sklearnCustom(Base):
    """Custom sklearn model derived from class Base
    
       Parameters
       ----------
       n_hidden_features: int
           number of nodes in the hidden layer
       activation_name: str
           activation function: 'relu', 'tanh' or 'sigmoid'
    """
    
    # construct the object -----
    
    def __init__(self, regr,
                 n_hidden_features=5, 
                 activation_name='relu',
                 nodes_sim='sobol',
                 n_clusters=None,
                 seed = 123):
                
        super().__init__(n_hidden_features, activation_name,
                         nodes_sim, n_clusters, seed)
        self.regr = regr

        
    def get_params(self):
        
        return super().get_params()

    
    def set_params(self, n_hidden_features=5, 
                   activation_name='relu', nodes_sim='sobol',
                   n_clusters=None, seed = 123):
        
        super().set_params(n_hidden_features=n_hidden_features, 
             activation_name=activation_name, nodes_sim=nodes_sim, 
             n_clusters=n_clusters, seed=seed)
 
    
    def fit(self, X, y, **kwargs):
        
        centered_y, scaled_Z = self.preproc_training_set(y = y, X = X)
        self.regr.fit(scaled_Z, centered_y, **kwargs)
        
        return self

    
    def predict(self, X, **kwargs):
        
        return self.y_mean + self.regr.predict(self.preproc_test_set(X), 
                                               **kwargs)

