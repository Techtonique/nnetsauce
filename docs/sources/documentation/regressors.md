# Documentation for regressors

_In alphabetical order_

loren ipsum 

<span style="float:right;">[[source]](https://github.com/Techtonique/nnetsauce/nnetsauce/base/baseRegressor.py#L14)</span>

### BaseRegressor


```python
nnetsauce.BaseRegressor(
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
)
```


Random Vector Functional Link Network regression without shrinkage 
 
Attributes
----------

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
    Currently available: standardization ('std') or MinMax scaling ('minmax')

col_sample: float
    percentage of features randomly chosen for training

row_sample: float
    percentage of rows chosen for training, by stratified bootstrapping   

seed: int 
    reproducibility seed for nodes_sim=='uniform', clustering and dropout

backend: str
    "cpu" or "gpu" or "tpu"                


----

<span style="float:right;">[[source]](https://github.com/Techtonique/nnetsauce/nnetsauce/custom/customRegressor.py#L12)</span>

### CustomRegressor


```python
nnetsauce.CustomRegressor(
    obj,
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
)
```


Custom Regression model 

Parameters
----------
obj: object
    any object containing a method fit (obj.fit()) and a method predict 
    (obj.predict())
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
    reproducibility seed for nodes_sim=='uniform'
type_fit: str
    'regression'
backend: str
    "cpu" or "gpu" or "tpu"                           


----

