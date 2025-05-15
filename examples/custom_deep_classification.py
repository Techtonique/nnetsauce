import os 
import nnetsauce as ns 
import numpy as np
import sklearn.metrics as skm2
from sklearn.datasets import load_breast_cancer, load_iris, load_wine, load_digits
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, RidgeClassifierCV, SGDClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from time import time 

print(f"\n ----- Running: {os.path.basename(__file__)}... ----- \n")

print("Example 1 - not conformal")

load_models = [load_breast_cancer, load_iris, load_wine]

for model in load_models: 

    data = model()
    X = data.data
    y= data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .3, random_state = 13)

    obj = SVC()

    clf = ns.DeepClassifier(obj, n_layers=2, verbose=1, n_clusters=2, n_hidden_features=2)

    start = time()
    clf.fit(X_train, y_train)
    print(f"\nElapsed: {time() - start} seconds\n")

    preds = clf.predict(X_test)

    print(clf.score(X_test, y_test))

print("Example 2 - not conformal with weights")

load_models = [load_breast_cancer, load_iris, load_wine]

for model in load_models: 

    data = model()
    X = data.data
    y= data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .3, random_state = 13)

    obj = SVC()

    clf = ns.DeepClassifier(obj, n_layers=2, verbose=1, n_clusters=2, n_hidden_features=2)

    start = time()
    clf.fit(X_train, y_train, sample_weight=np.random.rand(X_train.shape[0]))
    print(f"\nElapsed: {time() - start} seconds\n")

    preds = clf.predict(X_test)

    print(clf.score(X_test, y_test))

# print("Example 3 - conformal")

# for model in load_models: 

#     data = model()
#     X = data.data
#     y= data.target
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .3, random_state = 13)

#     obj = LogisticRegression()

#     clf = ns.DeepClassifier(obj, n_layers=2, verbose=1, 
#                             n_clusters=2, n_hidden_features=2, 
#                             level=95, pi_method="tcp")

#     start = time()
#     clf.fit(X_train, y_train)
#     print(f"\nElapsed: {time() - start} seconds\n")

#     print(f"clf: {clf}")

#     preds = clf.predict(X_test)

#     print(f"preds: {preds}")

#     print(f"accuracy:{np.mean(preds.argmax(axis=1) == y_test)}")