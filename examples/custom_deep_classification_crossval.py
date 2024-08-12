import joblib 
import numpy as np  
import os 
import pandas as pd 
import nnetsauce as ns 
import sklearn.metrics as skm2
from sklearn.datasets import load_breast_cancer, load_iris, load_wine, load_digits
from sklearn.linear_model import LogisticRegressionCV, RidgeClassifierCV, SGDClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from time import time 
from functools import partial
from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.utils import all_estimators


removed_classifiers = [
    "ClassifierChain",
    "ComplementNB",
    "GaussianProcessClassifier",
    "GradientBoostingClassifier",
    "HistGradientBoostingClassifier",
    "MultiOutputClassifier",
    "MultinomialNB",
    "OneVsOneClassifier",
    "OneVsRestClassifier",
    "OutputCodeClassifier",
    "RadiusNeighborsClassifier",
    "StackingClassifier",
    "VotingClassifier",
]

removed_regressors = [
    "TheilSenRegressor",
    "ARDRegression",
    "CCA",
    "GaussianProcessRegressor",
    "GradientBoostingRegressor",
    "HistGradientBoostingRegressor",
    "IsotonicRegression",
    "KernelRidge",
    "MultiOutputRegressor",
    "MultiTaskElasticNet",
    "MultiTaskElasticNetCV",
    "MultiTaskLasso",
    "MultiTaskLassoCV",
    "NuSVR",
    "OrthogonalMatchingPursuit",
    "OrthogonalMatchingPursuitCV",
    "PLSCanonical",
    "PLSRegression",
    "RadiusNeighborsRegressor",
    "RegressorChain",
    "StackingRegressor",
    "SVR",
    "VotingRegressor",
]

CLASSIFIERS = [
    (est[0], est[1])
    for est in all_estimators()
    if (
        issubclass(est[1], ClassifierMixin)
        and (est[0] not in removed_classifiers)
    )
]

SIMPLEMULTITASKCLASSIFIERS = [
    (
        "SimpleMultitaskClassifier(" + est[0] + ")",
        partial(ns.SimpleMultitaskClassifier, obj=est[1]()),
    )
    for est in all_estimators()
    if (
        issubclass(est[1], RegressorMixin)
        and (est[0] not in removed_regressors)
    )
]


print(f"\n ----- Running: {os.path.basename(__file__)}... ----- \n")

load_datasets = [load_wine, load_iris, load_breast_cancer, load_digits]

classifiers = CLASSIFIERS + SIMPLEMULTITASKCLASSIFIERS            

dataset_names = ["wine", "iris", "breast_cancer", "digits"]

model_names = []

results_df = pd.DataFrame(np.zeros((100, len(dataset_names))), columns=dataset_names)

results_df.index = ["None" for idx in range(100)]

for j, dataset in enumerate(load_datasets): 

    print(f"\n\n {j+1}/{len(dataset_names)} - dataset name: {dataset_names[j]} ---------- \n")

    data = dataset()
    X = data.data
    y= data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .2, 
                                                        random_state = 123)

    for i, obj in enumerate(classifiers):
        
        try: 

            clf_name = type(obj[1]()).__name__

            print(f"\n {i+1}/{len(classifiers)} - base learner: {clf_name} ----------")

            clf = ns.DeepClassifier(obj=obj[1]())
            
            res = clf.cross_val_optim(X_train, y_train, X_test, y_test, verbose=1)

            print(res)

            results_df.iloc[i, j] = res.test_accuracy
            results_df.index[i] = clf_name 
        
        except: 

            pass 

print(results_df)

joblib.dump(results_df, 'results_df.pkl')