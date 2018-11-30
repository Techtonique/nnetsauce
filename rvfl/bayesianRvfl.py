from ..base import Base
from ..utils import misc as mx


class BayesianRvfl(Base):
    """Bayesian RVFL model class derived from class Base
    
       Parameters
       ----------
       obj: object
           any object containing a method fit (obj.fit()) and a method predict (obj.predict())
       n_hidden_features: int
           number of nodes in the hidden layer
       activation_name: str
           activation function: 'relu', 'tanh' or 'sigmoid'
       nodes_sim: str
           type of simulation for the nodes: 'sobol', 'hammersley', 'halton', 'uniform'
       bias: boolean
           indicates if the hidden layer contains a bias term (True) or not (False)
       direct_link: boolean
           indicates if the original predictors are included (True) in model's fitting or not (False)
       n_clusters: int
           number of clusters for 'kmeans' or 'gmm' clustering (could be 0: no clustering)
       type_clust: str
           type of clustering method: currently k-means ('kmeans') or Gaussian Mixture Model ('gmm')
       seed: int 
           reproducibility seed for nodes_sim=='uniform'
    """
    
    # construct the object -----
    
    def __init__(self,
                 n_hidden_features=5, 
                 activation_name='relu',
                 nodes_sim='sobol',
                 bias=True,
                 direct_link=True, 
                 n_clusters=2,
                 type_clust='kmeans',
                 seed=123, 
                 s=0.1, sigma=0.05):
                
        super().__init__(n_hidden_features, activation_name,
                         nodes_sim, bias, direct_link,
                         n_clusters, type_clust, seed)
        self.s = s 
        self.sigma = sigma
    
    
    def get_params(self):
        
        return mx.merge_two_dicts(super().get_params(), 
                                  {"s": self.s, 
                                   "sigma": self.sigma})

    
    def set_params(self, n_hidden_features=5, 
                   activation_name='relu', 
                   nodes_sim='sobol',
                   bias=True,
                   direct_link=True,
                   n_clusters=None,
                   type_clust='kmeans',
                   seed=123, 
                   s=0.1, sigma=0.05):
        
        super().set_params(n_hidden_features=n_hidden_features, 
                           activation_name=activation_name, nodes_sim=nodes_sim,
                           bias=bias, direct_link=direct_link, 
                           n_clusters=n_clusters, type_clust=type_clust, 
                           seed=seed, s=s, sigma=sigma)