import os 
import nnetsauce as ns 
import numpy as np 
from sklearn.datasets import load_diabetes, load_iris, load_wine
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor, SGDClassifier, Perceptron
from sklearn.gaussian_process.kernels import Matern
from time import time 

print(f"\n ----- Running: {os.path.basename(__file__)}... ----- \n")

print(f"\n\n ----- iris ----- \n\n")

data = load_iris()

X = data.data
y= data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .2, random_state = 123)


regr = ns.CustomClassifier(SGDClassifier(), n_clusters=0)
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


print(f"\n\n ----- diabetes ----- \n\n")

data = load_diabetes()
X = data.data
y= data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .2, random_state = 123)


regr = ns.CustomRegressor(SGDRegressor(), n_clusters=0, n_hidden_features=10)
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

