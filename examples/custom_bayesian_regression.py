import os 
import nnetsauce as ns 
import numpy as np 
from sklearn.datasets import load_diabetes, fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import BayesianRidge, ARDRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from time import time 

print(f"\n ----- Running: {os.path.basename(__file__)}... ----- \n")

print(f"\n ----- diabetes ----- \n")

data = load_diabetes()
X = data.data
y= data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .2, random_state = 123)

print(X.shape)
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

regr5 = ns.CustomRegressor(BayesianRidge())
start = time()
regr5.fit(X_train, y_train)
print(f"Elapsed: {time() - start}s")
preds5 = regr5.predict(X_test, return_pi=True)
print(f"coverage_rate conformalized Bayesian Ridge: {np.mean((preds5[1]<=y_test)*(preds5[2]>=y_test))}")
regr5.fit(X_train, y_train)
preds5 = regr5.predict(X_test, method="localconformal", return_pi=True)
print(f"coverage_rate conformalized Bayesian Ridge: {np.mean((preds5[1]<=y_test)*(preds5[2]>=y_test))}")

regr6 = ns.CustomRegressor(ARDRegression())
start = time()
regr6.fit(X_train, y_train)
print(f"Elapsed: {time() - start}s")
preds6 = regr6.predict(X_test, return_pi=True)
print(f"coverage_rate conformalized ARD Regressor: {np.mean((preds6[1]<=y_test)*(preds6[2]>=y_test))}")


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
regr3.fit(X_train, y_train)
preds3 = regr3.predict(X_test, method="localconformal", return_std=True)
print(f"coverage_rate Gaussian Process: {np.mean((preds3[2]<=y_test)*(preds3[3]>=y_test))}")

print(f"\n ----- california housing ----- \n")

data = fetch_california_housing()
X = data.data
y= data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .2, random_state = 123)

print(X.shape)
regr = ns.CustomRegressor(BayesianRidge())
start = time()
regr.fit(X_train, y_train)
print(f"Elapsed: {time() - start}s")
preds = regr.predict(X_test, return_std=True)
print(f"coverage_rate Bayesian Ridge: {np.mean((preds[2]<=y_test)*(preds[3]>=y_test))}")

regr3 = ns.CustomRegressor(BayesianRidge())
start = time()
regr3.fit(X_train, y_train)
print(f"Elapsed: {time() - start}s")
regr3.fit(X_train, y_train)
preds3 = regr3.predict(X_test, return_pi=True)
print(f"coverage_rate conformalized Bayesian Ridge: {np.mean((preds3[1]<=y_test)*(preds3[2]>=y_test))}")

regr2 = ns.CustomRegressor(ARDRegression())
start = time()
regr2.fit(X_train, y_train)
print(f"Elapsed: {time() - start}s")
preds2 = regr2.predict(X_test, return_std=True)
print(f"preds2: {preds2}")
print(f"coverage_rate ARD Regressor: {np.mean((preds2[2]<=y_test)*(preds2[3]>=y_test))}")

regr4 = ns.CustomRegressor(ARDRegression())
start = time()
regr4.fit(X_train, y_train)
print(f"Elapsed: {time() - start}s")
preds4 = regr4.predict(X_test, return_pi=True, level=90)
print(f"coverage_rate conformalized ARD Regressor: {np.mean((preds4[1]<=y_test)*(preds4[2]>=y_test))}")
regr4.fit(X_train, y_train)
start = time()
preds4 = regr4.predict(X_test, method="localconformal", return_pi=True, level=90)
print(f"Elapsed: {time() - start}s")
print(f"preds4: {preds4}")
print(f"coverage_rate conformalized ARD Regressor: {np.mean((preds4[1]<=y_test)*(preds4[2]>=y_test))}")
