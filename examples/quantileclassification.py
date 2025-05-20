import nnetsauce as ns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, Lasso, ElasticNet, RidgeCV, LassoCV, ElasticNetCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.datasets import load_iris, load_breast_cancer, load_wine, load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm


scoring = ["conformal", "residuals", "predictions", "studentized", "conformal-studentized"]

datasets = [load_iris, load_breast_cancer, load_wine, load_digits]

dataset_names = ["iris", "breast_cancer", "wine", "digits"]

regrs = [RidgeCV(), LassoCV(), KNeighborsRegressor(), RandomForestRegressor()] 

for dataset, dataset_name in zip(datasets, dataset_names):

    print("\n\n dataset", dataset_name, "--------------------------------")

    X, y = dataset(return_X_y=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                        random_state=42)

    for score in tqdm(scoring):

        print("\n score", score)

        for regr in regrs:

            print("\n regr", regr.__class__.__name__)

            classifier = ns.QuantileClassifier(
                obj=regr, 
                scoring = score   
            )

            classifier.fit(X_train, y_train)
            predictions_proba = classifier.predict_proba(X_test)
            
            predictions = classifier.predict(X_test)

            print("score", classification_report(y_test, predictions))
