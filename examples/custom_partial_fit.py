import os 
import nnetsauce as ns 
import numpy as np 
from sklearn.datasets import load_diabetes, load_breast_cancer, load_iris, load_wine, fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor, SGDClassifier, Perceptron, PassiveAggressiveClassifier, PassiveAggressiveRegressor
from sklearn.gaussian_process.kernels import Matern
from time import time 


print(f"\n ----- Running: {os.path.basename(__file__)}... ----- \n")

datasets = [load_diabetes(), fetch_california_housing()]

dataset_names = ["diabetes", "california_housing"]

print("\n\n regressors")

for data, name in zip(datasets, dataset_names):

    X = data.data
    y= data.target
    if name == "diabetes":
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .2, random_state = 123)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X[:1000], y[:1000], test_size = .2, random_state = 123)

    regr = ns.CustomRegressor(PassiveAggressiveRegressor(), n_clusters=0, n_hidden_features=100)
    start = time()
    regr.fit(X_train, y_train)
    print(f"Elapsed: {time() - start}s")
    preds = regr.predict(X_test)
    print(np.sqrt(np.mean((preds - y_test)**2)))
    regr.partial_fit(X_test[0, :], y_test[0])
    preds = regr.predict(X_test[1:, :])
    print(np.sqrt(np.mean((preds - y_test[1:])**2)))
    regr.partial_fit(X_test[1, :], y_test[1])
    preds = regr.predict(X_test[2:, :])
    print(np.sqrt(np.mean((preds - y_test[2:])**2)))
    preds = regr.predict(X_test[3:, :])
    print(np.sqrt(np.mean((preds - y_test[3:])**2)))


print("\n\n classifiers")

datasets = [load_iris(), load_wine(), load_breast_cancer()]

dataset_names = ["iris", "wine", "breast_cancer"]

for data, name in zip(datasets, dataset_names):    

    print(f"\n\n ----- {name} ----- \n\n")

    X = data.data
    y= data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .2, random_state = 123)


    regr = ns.CustomClassifier(PassiveAggressiveClassifier(), 
                               n_clusters=0, 
                               n_hidden_features=100, 
                               cv_calibration=None) # as 'CalibratedClassifierCV' object has no attribute 'partial_fit'
    start = time()
    regr.fit(X_train, y_train)
    print(f"Elapsed: {time() - start}s")
    preds = regr.predict(X_test)
    print(regr.score(X_test, y_test))
    regr.partial_fit(X_test[0, :], y_test[0])
    preds = regr.predict(X_test[1:, :])
    print(regr.score(X_test[1:, :], y_test[1:]))
    regr.partial_fit(X_test[1, :], y_test[1])
    preds = regr.predict(X_test[2:, :])
    print(regr.score(X_test[2:, :], y_test[2:]))
    preds = regr.predict(X_test[3:, :])
    print(regr.score(X_test[3:, :], y_test[3:]))


