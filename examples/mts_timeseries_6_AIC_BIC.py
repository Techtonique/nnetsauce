import os 
import nnetsauce as ns
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn import datasets, metrics
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.base.datetools import dates_from_str

print(f"\n ----- Running: {os.path.basename(__file__)}... ----- \n")

# Example 1 

np.random.seed(1235)
M = np.random.rand(100, 3)
M[:,0] = 10*M[:,0]
M[:,2] = 25*M[:,2]

print(M)
print("\n")

# Adjust Ridge
regr4 = linear_model.Ridge()
obj_MTS = ns.MTS(regr4, 
                 lags = 1, 
                 n_hidden_features=5, 
                 verbose = 1, 
                 type_pi="gaussian", 
                 auto_lags="AIC")
obj_MTS.fit(M)

# with Gaussian prediction intervals
res1 = obj_MTS.predict(level=95) 
print(res1)
print("\n")

# Example 2 

# Adjust Ridge
regr4 = linear_model.Ridge()
obj_MTS = ns.MTS(regr4, 
                 lags = 1, 
                 n_hidden_features=5, 
                 verbose = 1, 
                 type_pi="gaussian", 
                 auto_lags="AIC")
obj_MTS.fit(M)

# with Gaussian prediction intervals
res1 = obj_MTS.predict(level=95) 
print(res1)
print("\n")
