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
res = clf.cross_val_optim(X_train, y_train, X_test, y_test, 
                          abs_tol=10**-2, 
                          n_jobs=5, scoring="balanced_accuracy",
                          surrogate_obj=GaussianProcessRegressor(
                                kernel=Matern(nu=2.5),
                                alpha=1e-6,
                                normalize_y=True,
                                n_restarts_optimizer=25,
                                random_state=123,
                            ),
                        verbose=1)
print(res)


# for i, row_name in enumerate(row_names):
#     for j, dataset in enumerate(load_datasets): 
#         print(f"\n data set #{j+1}/{len(load_datasets)} - tol: {i+1}/{len(row_names)} - data set: {col_names[j]} ----------")
#         data = dataset()
#         X = data.data
#         y= data.target
#         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .2, 
#                                                             random_state = 123)
#         clf = ns.DeepClassifier(obj=ns.SimpleMultitaskClassifier(ExtraTreesRegressor()))            
#         res = clf.cross_val_optim(X_train, y_train, X_test, y_test, 
#                                 abs_tol=row_names[i], 
#                                 n_jobs=5, scoring="balanced_accuracy",
#                                 verbose=1)
#         print(res)
#         df_balanced_accuracy.iloc[i, j] = res.test_balanced_accuracy

# print(df_balanced_accuracy)

# joblib.dump(df_balanced_accuracy, 'df_balanced_accuracy.pkl')

# for i, row_name in enumerate(row_names):
#     for j, dataset in enumerate(load_datasets): 
#         print(f"\n data set #{j+1}/{len(load_datasets)} - tol: {i+1}/{len(row_names)} ({row_names[i]}) - data set: {col_names[j]} ----------")
#         data = dataset()
#         X = data.data
#         y= data.target
#         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .2, 
#                                                             random_state = 123)
#         clf = ns.DeepClassifier(obj=ns.SimpleMultitaskClassifier(ExtraTreesRegressor()))            
#         res = clf.cross_val_optim(X_train, y_train, X_test, y_test, 
#                                 abs_tol=row_names[i], 
#                                 n_jobs=5, scoring="balanced_accuracy",
#                                 surrogate_obj=GaussianProcessRegressor(
#                                         kernel=Matern(nu=2.5),
#                                         alpha=1e-6,
#                                         normalize_y=True,
#                                         n_restarts_optimizer=25,
#                                         random_state=123,
#                                     ),
#                                 verbose=1)
#         print(res)
#         #df_balanced_accuracy2.iloc[i, j] = res.test_balanced_accuracy

# print(df_balanced_accuracy2)

# #joblib.dump(df_balanced_accuracy2, 'df_balanced_accuracy2.pkl')
