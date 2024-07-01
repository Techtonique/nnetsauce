import os 
import nnetsauce as ns
import numpy as np
import pandas as pd
from sklearn import datasets, metrics
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor

print(f"\n ----- Running: {os.path.basename(__file__)}... ----- \n")

np.random.seed(1235)


M = np.random.rand(10, 3)
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
                 type_pi="gaussian")
obj_MTS.fit(M)

# with Gaussian prediction intervals
res1 = obj_MTS.predict(level=80) 
print(res1)
print("\n")


