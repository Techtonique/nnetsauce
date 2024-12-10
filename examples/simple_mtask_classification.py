import os 
import nnetsauce as ns
import numpy as np
from sklearn.datasets import load_breast_cancer, load_wine, load_digits, load_iris, make_classification
from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn import metrics
from time import time

print(f"\n ----- Running: {os.path.basename(__file__)}... ----- \n")

# dataset no. 1 ---------- 

print(" \n breast cancer dataset ----- \n")

breast_cancer = load_breast_cancer()
Z = breast_cancer.data
t = breast_cancer.target

X_train, X_test, y_train, y_test = train_test_split(Z, t, test_size=0.2, 
                                                    random_state=123+2*10)

# Linear Regression is used 
regr = RandomForestRegressor()
fit_obj = ns.SimpleMultitaskClassifier(regr)

start = time()
fit_obj.fit(X_train, y_train)
print(f"Elapsed {time() - start}") 

print(fit_obj.score(X_test, y_test))

start = time()
preds = fit_obj.predict(X_test)
print(f"Elapsed {time() - start}") 
print(metrics.classification_report(preds, y_test))

# dataset no. 4 ---------- 

print(" \n wine dataset ----- \n")

dataset = load_wine()
Z = dataset.data
t = dataset.target
#np.random.seed(123)
X_train, X_test, y_train, y_test = train_test_split(Z, t, test_size=0.2, 
                                                    random_state=123+2*10)

# Linear Regression is used 
regr = RandomForestRegressor()
fit_obj = ns.SimpleMultitaskClassifier(regr)

start = time()
fit_obj.fit(X_train, y_train)
print(f"Elapsed {time() - start}") 

print(fit_obj.score(X_test, y_test))

start = time()
preds = fit_obj.predict(X_test)
print(f"Elapsed {time() - start}") 
print(metrics.classification_report(preds, y_test))

# dataset no. 3 ---------- 

print(" \n iris dataset ----- \n")

dataset = load_iris()
Z = dataset.data
t = dataset.target
#np.random.seed(123)
X_train, X_test, y_train, y_test = train_test_split(Z, t, test_size=0.2, 
                                                    random_state=123+2*10)

# Linear Regression is used 
regr = RandomForestRegressor()
fit_obj = ns.SimpleMultitaskClassifier(regr)

start = time()
fit_obj.fit(X_train, y_train)
print(f"Elapsed {time() - start}") 

print(fit_obj.score(X_test, y_test))

start = time()
preds = fit_obj.predict(X_test)
print(f"Elapsed {time() - start}") 
print(metrics.classification_report(preds, y_test))

# dataset no. 2 ---------- 

print(" \n digits dataset ----- \n")

dataset = load_digits()
Z = dataset.data
t = dataset.target
#np.random.seed(123)
X_train, X_test, y_train, y_test = train_test_split(Z, t, test_size=0.2, 
                                                    random_state=123+2*10)

# Linear Regression is used 
regr = RandomForestRegressor()
fit_obj = ns.SimpleMultitaskClassifier(regr)

start = time()
fit_obj.fit(X_train, y_train)
print(f"Elapsed {time() - start}") 

print(fit_obj.score(X_test, y_test))

start = time()
preds = fit_obj.predict(X_test)
print(f"Elapsed {time() - start}") 
print(metrics.classification_report(preds, y_test))

