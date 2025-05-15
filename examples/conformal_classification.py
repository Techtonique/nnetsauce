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

load_models = [load_iris, load_breast_cancer, load_wine]

for model in load_models: 

    data = model()
    X = data.data
    y= data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .2, random_state = 13)

    print("\n\n CustomClassifier -----")
    obj = LogisticRegression()
    clf = ns.CustomClassifier(obj, level=95, pi_method="icp")

    start = time()
    clf.fit(X_train, y_train)
    print(f"\nElapsed: {time() - start} seconds\n")

    preds_proba = clf.predict_proba(X_test)
    print(f"accuracy: {np.mean(preds_proba.argmax(axis=1) == y_test)}")    

    print("\n\n DeepClassifier -----")
    obj = LogisticRegression()
    clf = ns.DeepClassifier(obj, level=95, pi_method="icp")

    start = time()
    clf.fit(X_train, y_train)
    print(f"\nElapsed: {time() - start} seconds\n")

    preds_proba = clf.predict_proba(X_test)
    print(f"accuracy: {np.mean(preds_proba.argmax(axis=1) == y_test)}")    
