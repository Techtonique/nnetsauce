import os 
import nnetsauce as ns 
import numpy as np 
from sklearn.datasets import load_diabetes, fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import BayesianRidge, ARDRegression
from sklearn.ensemble import ExtraTreesRegressor
from time import time 

print(f"\n ----- Running: {os.path.basename(__file__)}... ----- \n")

print(f"\n ----- fetch_california_housing ----- \n")

data = fetch_california_housing()
X = data.data
y= data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .2, random_state = 13)
print(f"X_train.shape(): {X_train.shape}")
print(f"X_test.shape(): {X_test.shape}")
regr = ns.PredictionInterval(obj=ExtraTreesRegressor(),
                             type_pi="bootstrap",
                             replications=100, 
                             level=80,
                             seed=12)
start = time()
regr.fit(X_train, y_train)
print(f"Elapsed: {time() - start}s")
preds = regr.predict(X_test)
print(preds)
print(f"coverage_rate: {np.mean((preds[2]<=y_test)*(preds[3]>=y_test))}")

