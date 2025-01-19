import os 
import nnetsauce as ns
import numpy as np
import pandas as pd
from sklearn import datasets, metrics
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor

print(f"\n ----- Running: {os.path.basename(__file__)}... ----- \n")

np.random.seed(1235)


M = np.random.rand(100, 3)
M[:,0] = 10*M[:,0]
M[:,2] = 25*M[:,2]

print(M)
print("\n")

# Adjust Bayesian Ridge
regr4 = linear_model.BayesianRidge()
obj_MTS = ns.MTS(regr4, lags = 1, n_hidden_features=5, verbose = 1)
obj_MTS.fit(M)

# with credible intervals
res1 = obj_MTS.predict(return_std=True, level=80) 
print(res1)
print("\n")


# Adjust Bayesian Ridge
obj_MTS2 = ns.MTS(obj=ns.CustomRegressor(obj=linear_model.Ridge()), 
                  lags = 1, 
                  n_hidden_features=5, 
                  verbose = 1)
obj_MTS2.fit(M)

# with conformal prediction
res2 = obj_MTS2.predict(return_pi=True, level=80, method="splitconformal") 
print(res2)
print("\n")

