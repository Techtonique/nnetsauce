import joblib
import os 
import numpy as np 
import pandas as pd 
import nnetsauce as ns 
import sklearn.metrics as skm2
from sklearn.datasets import load_breast_cancer, load_iris, load_wine, load_digits
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.model_selection import train_test_split
from time import time 

print(f"\n ----- Running: {os.path.basename(__file__)}... ----- \n")

load_datasets = [load_breast_cancer, load_iris, load_wine, load_digits]
col_names = ["breast_cancer", "iris", "wine", "digits"]
row_names = [10**-i for i in range(1, 6)]
df_balanced_accuracy = pd.DataFrame(np.zeros((len(row_names), len(col_names))), 
                                    columns=col_names)
df_balanced_accuracy2 = pd.DataFrame(np.zeros((len(row_names), len(col_names))), 
                                    columns=col_names)
df_balanced_accuracy.index = row_names 
df_balanced_accuracy2.index = row_names 

data = load_datasets[1]()
X = data.data
y= data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .2, 
                                                    random_state = 123)
clf = ns.DeepClassifier(obj=ns.SimpleMultitaskClassifier(ExtraTreesRegressor()))            
res = clf.lazy_cross_val_optim(X_train, y_train, X_test, y_test, 
                               abs_tol=10**-2, 
                               n_jobs=5, 
                               surrogate_objs=["RidgeCV", "ElasticNetCV",
                                                "LarsCV", "LassoCV", "LassoLarsCV", 
                                                "LassoLarsIC"],
                               customize=True,
                               scoring="balanced_accuracy",                         
                               verbose=1)
print(res)
