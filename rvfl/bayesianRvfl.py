from ..base import Base


class bayesianRvfl(Base):
    """Bayesian RVFL model derived from class Base
    
       Parameters
       ----------
       n_hidden_features: int
           number of nodes in the hidden layer
       activation_name: str
           activation function: 'relu', 'tanh' or 'sigmoid'
    """
    
    # construct the object -----
    
    def __init__(self,
                 n_hidden_features=5, 
                 activation_name='relu',
                 nodes_sim='sobol',
                 n_clusters=None,
                 seed = 123,
                 s_beta = 0.1,
                 sigma = 0.5):
                
        super().__init__(n_hidden_features, activation_name,
                         nodes_sim, n_clusters, seed)
        self.s_beta = s_beta
        self.sigma = sigma