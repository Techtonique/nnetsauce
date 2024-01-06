import os 
import nnetsauce as ns 
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

for model in load_models:

    obj = model()

    regr = ns.DeepRegressor(obj, n_layers=3, verbose=1, n_clusters=3, n_hidden_features=2)

    start = time()
    regr.fit(X_train, y_train)
    print(f"\nElapsed: {time() - start} seconds\n")

    print(f"{type(obj).__name__} test RMSE: {regr.score(X_test, y_test)} \n")
