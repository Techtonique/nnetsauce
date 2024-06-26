import os 
import nnetsauce as ns 
import numpy as np 
from sklearn.datasets import load_diabetes, fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from time import time 

print(f"\n ----- Running: {os.path.basename(__file__)}... ----- \n")

print(f"\n ----- fetch_california_housing ----- \n")

data = fetch_california_housing()
X = data.data
y= data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .3, random_state = 213)
print(f"X_train.shape(): {X_train.shape}")
print(f"X_test.shape(): {X_test.shape}")

regr = ns.PredictionInterval(obj=ExtraTreesRegressor(),
                             method="splitconformal",
                             level=95,
                             seed=312)
start = time()
regr.fit(X_train, y_train)
print(f"Elapsed: {time() - start}s")
preds = regr.predict(X_test, return_pi=True)
print(preds)
print(f"coverage_rate: {np.mean((preds.lower<=y_test)*(preds.upper>=y_test))}")

regr3 = ns.PredictionInterval(obj=ExtraTreesRegressor(),
                             method="splitconformal",
                             type_split="sequential",
                             level=95,
                             seed=312)
start = time()
regr3.fit(X_train, y_train)
print(f"Elapsed: {time() - start}s")
preds = regr3.predict(X_test, return_pi=True)
print(preds)
print(f"coverage_rate: {np.mean((preds.lower<=y_test)*(preds.upper>=y_test))}")

