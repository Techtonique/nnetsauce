import os 
import nnetsauce as ns 
import numpy as np 
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import BayesianRidge, ARDRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from time import time 

print(f"\n ----- Running: {os.path.basename(__file__)}... ----- \n")

data = load_diabetes()
X = data.data
y= data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .2, random_state = 123)

regr = ns.CustomRegressor(BayesianRidge())
start = time()
regr.fit(X_train, y_train)
print(f"Elapsed: {time() - start}s")
preds = regr.predict(X_test, return_std=True)
print(f"coverage_rate Bayesian Ridge: {np.mean((preds[2]<=y_test)*(preds[3]>=y_test))}")

regr2 = ns.CustomRegressor(ARDRegression())
start = time()
regr2.fit(X_train, y_train)
print(f"Elapsed: {time() - start}s")
preds2 = regr2.predict(X_test, return_std=True)
print(f"coverage_rate ARD Regressor: {np.mean((preds2[2]<=y_test)*(preds2[3]>=y_test))}")

regr3 = ns.CustomRegressor(GaussianProcessRegressor(kernel=Matern(nu=1.5),
                          alpha=1e-6,
                          normalize_y=True,
                          n_restarts_optimizer=25,
                          random_state=42,))
start = time()
regr3.fit(X_train, y_train)
print(f"Elapsed: {time() - start}s")
preds3 = regr3.predict(X_test, return_std=True)
print(f"coverage_rate Gaussian Process: {np.mean((preds3[2]<=y_test)*(preds3[3]>=y_test))}")