import nnetsauce as ns
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, metrics
from sklearn import linear_model
np.random.seed(123)

M = np.random.rand(10, 3)
M[:,0] = 10*M[:,0]
M[:,2] = 25*M[:,2]
print(M)
print("\n")

# Adjust Bayesian Ridge
regr4 = linear_model.BayesianRidge()
obj_MTS = ns.MTS(regr4, lags = 1, n_hidden_features=5)
obj_MTS.fit(M)
print(obj_MTS.predict())
print("\n")

# with credible intervals
print(obj_MTS.predict(return_std=True, level=80))
print("\n")

print(obj_MTS.predict(return_std=True, level=95))
print("\n")

