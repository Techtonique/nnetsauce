# Regression example

Other examples can be found here: [https://thierrymoudiki.github.io/blog/#QuasiRandomizedNN](https://thierrymoudiki.github.io/blog/#QuasiRandomizedNN)

```python

import nnetsauce as ns
import numpy as np      
import matplotlib.pyplot as plt

from sklearn import datasets, metrics
from sklearn import linear_model, gaussian_process

# load datasets
diabetes = datasets.load_diabetes()
X = diabetes.data 
y = diabetes.target

index_train = range(100)
index_test = range(100, 125)

# layer 1 (base layer) ----
layer1_regr = linear_model.BayesianRidge()
layer1_regr.fit(X[index_train,:], y[index_train])

# RMSE score on test set
print(np.sqrt(metrics.mean_squared_error(y[index_test], layer1_regr.predict(X[index_test,:]))))


# layer 2 using layer 1 ----
layer2_regr = ns.CustomRegressor(obj = layer1_regr, n_hidden_features=3, 
                        direct_link=True, bias=True, 
                        nodes_sim='sobol', activation_name='tanh', 
                        n_clusters=2)
layer2_regr.fit(X[index_train,:], y[index_train])

# RMSE score on test set
print(np.sqrt(layer2_regr.score(X[index_test,:], y[index_test])))

# layer 3 using layer 2 ----
layer3_regr = ns.CustomRegressor(obj = layer2_regr, n_hidden_features=5, 
                        direct_link=True, bias=True, 
                        nodes_sim='hammersley', activation_name='sigmoid', 
                        n_clusters=2)
layer3_regr.fit(X[index_test,:], y[index_test])

# RMSE score on test set
print(np.sqrt(layer3_regr.score(X[index_test,:], y[index_test])))

```