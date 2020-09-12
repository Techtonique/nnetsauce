# Time series forecasting example

Other examples can be found here: [https://thierrymoudiki.github.io/blog/#QuasiRandomizedNN](https://thierrymoudiki.github.io/blog/#QuasiRandomizedNN)

```python

import nnetsauce as ns
import numpy as np      
import matplotlib.pyplot as plt
from sklearn import datasets, metrics
from sklearn import linear_model, gaussian_process
np.random.seed(123)

M = np.random.rand(10, 3)
M[:,0] = 10*M[:,0]
M[:,2] = 25*M[:,2]
print(M)

regr4 = gaussian_process.GaussianProcessRegressor()
obj_MTS = ns.MTS(regr4, lags = 1, n_hidden_features=5, 
                 bias = False)
obj_MTS.fit(M)
print(obj_MTS.predict())


# With a deep stack of 'Custom' objects (from previous snippet)
obj_MTS2 = ns.MTS(layer3_regr, lags = 1, n_hidden_features=5, 
                 bias = False)
obj_MTS2.fit(M)
print(obj_MTS2.predict())


# Choosing different scalings for the input variables (first input
# of tuple 'type_scaling') , hidden layer (second input
# of tuple 'type_scaling'), and clustering (third input
# of tuple 'type_scaling'). 
# This is also available for models Base, Custom, etc.

# 'minmax', 'minmax', 'std' scalings
regr6 = linear_model.BayesianRidge()
obj_MTS3 = ns.MTS(regr6, lags = 1, n_hidden_features=2, 
                 bias = True, type_scaling = ('minmax', 'minmax', 'std'))
obj_MTS3.fit(M)
print(obj_MTS3.predict())

# 'minmax', 'standardization', 'minmax' scalings
regr7 = linear_model.BayesianRidge()
obj_MTS4 = ns.MTS(regr6, lags = 1, n_hidden_features=2, 
                 bias = True, type_scaling = ('minmax', 'std', 'minmax'))
obj_MTS4.fit(M)
print(obj_MTS4.predict())

```