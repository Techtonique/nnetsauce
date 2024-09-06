import os 
import nnetsauce as ns 
import numpy as np
import sklearn.metrics as skm2
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LassoCV, RidgeCV, ElasticNetCV, HuberRegressor
from sklearn.model_selection import train_test_split
from time import time 

print(f"\n ----- Running: {os.path.basename(__file__)}... ----- \n")

load_models = [LassoCV, RidgeCV, ElasticNetCV, HuberRegressor]

data = load_diabetes()
X = data.data
y= data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .2, random_state = 13)

print("\n\n 1 - not conformal \n\n")

for model in load_models:

    obj = model()

    regr = ns.DeepRegressor(obj, n_layers=3, verbose=1, n_clusters=3, n_hidden_features=2)

    start = time()
    regr.fit(X_train, y_train)
    print(f"\nElapsed: {time() - start} seconds\n")

    print(f"{type(obj).__name__} test RMSE: {regr.score(X_test, y_test)} \n")

print("\n\n 2 - conformal \n\n")

for model in load_models:

    obj = model()

    regr = ns.DeepRegressor(obj, n_layers=3, 
                            verbose=1, n_clusters=2, 
                            n_hidden_features=5, 
                            level=95, 
                            pi_method="splitconformal")

    start = time()
    regr.fit(X_train, y_train)
    print(f"\nElapsed: {time() - start} seconds\n")

    preds = regr.predict(X_test)
    print(f"preds: {preds}")
    coverage = np.mean((y_test >= preds.lower) & (y_test <= preds.upper))
    print(f"test coverage: {coverage} \n")
