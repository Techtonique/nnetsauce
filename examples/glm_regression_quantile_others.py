import os 
import GPopt as gp
import nnetsauce as ns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from time import time

print(f"\n ----- Running: {os.path.basename(__file__)}... ----- \n")

X, y = fetch_california_housing(return_X_y=True)    
X, y = X[:1500], y[:1500]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=2020)

opt = ns.Optimizer(type_optim="scd", 
                    learning_rate=0.1, 
                    batch_prop=0.25, 
                    verbose=0)
obj = ns.GLMRegressor(n_hidden_features=3, 
                       lambda1=1e-2, alpha1=0.5,
                       lambda2=1e-2, alpha2=0.5, 
                       family="quantile",
                       level=95,
                       optimizer=opt)

start = time()
obj.fit(X_train, y_train)
print(f"\n Elapsed: {time() - start}")
pred_upper = obj.predict(X_test)
print(f"pred_upper: {pred_upper}")

obj = ns.GLMRegressor(n_hidden_features=3, 
                       lambda1=1e-2, alpha1=0.5,
                       lambda2=1e-2, alpha2=0.5, 
                       family="quantile",
                       level=5,
                       optimizer=opt)

start = time()
obj.fit(X_train, y_train)
print(f"\n Elapsed: {time() - start}")
pred_lower = obj.predict(X_test)
print(f"pred_lower: {pred_lower}")

print(f"Check <=: {pred_lower <= pred_upper}")

print(f"Check - : {pred_upper - pred_lower}")

print(f"Coverage: {100*np.mean((y_test <= pred_upper)*(y_test >= pred_lower))}")