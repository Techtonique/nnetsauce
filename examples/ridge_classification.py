#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 14:29:02 2019

@author: moudiki
"""

import nnetsauce as ns
import numpy as np
from sklearn.datasets import load_digits, load_breast_cancer, load_wine, load_iris
from sklearn.model_selection import train_test_split


# dataset no. 1 ----------

breast_cancer = load_breast_cancer()
X = breast_cancer.data
y = breast_cancer.target

# split data into training test and test set
np.random.seed(123)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# create the model with nnetsauce
fit_obj = ns.Ridge2Classifier(lambda1 = 6.90185578e+04, 
                             lambda2 = 3.17392781e+02, 
                             n_hidden_features=95, 
                             n_clusters=2, 
                             row_sample = 4.63427734e-01, 
                             dropout = 3.62817383e-01,
                             type_clust = "gmm")

# fit the model on training set
start = time()
fit_obj.fit(X_train, y_train)
print(time() - start)

# get the accuracy on test set
start = time()
print(fit_obj.score(X_test, y_test))
print(time() - start)

# get area under the curve on test set (auc)
print(fit_obj.score(X_test, y_test, scoring="roc_auc"))


# dataset no. 2 ----------

wine = load_wine()
Z = wine.data
t = wine.target
np.random.seed(123)
Z_train, Z_test, y_train, y_test = train_test_split(Z, t, test_size=0.2)

# create the model with nnetsauce
fit_obj = ns.Ridge2Classifier(lambda1 = 8.64135756e+04, 
                             lambda2 = 8.27514666e+04, 
                             n_hidden_features=109, 
                             n_clusters=3, 
                             row_sample = 5.54907227e-01, 
                             dropout = 1.84484863e-01,
                             type_clust = "gmm")

# fit the model on training set
fit_obj.fit(Z_train, y_train)

# get the accuracy on test set
print(fit_obj.score(Z_test, y_test))


# dataset no. 3 ----------

iris = load_iris()
Z = iris.data
t = iris.target
np.random.seed(123)
Z_train, Z_test, y_train, y_test = train_test_split(Z, t, test_size=0.2)

# create the model with nnetsauce
fit_obj = ns.Ridge2Classifier(lambda1 = 1.87500081e+04, 
                             lambda2 = 3.12500069e+04, 
                             n_hidden_features=47, 
                             n_clusters=3, 
                             row_sample = 7.37500000e-01, 
                             dropout = 1.31250000e-01,
                             type_clust = "gmm")

# fit the model on training set
start = time()
fit_obj.fit(Z_train, y_train)
print(time() - start)

# get the accuracy on test set
start = time()
print(fit_obj.score(Z_test, y_test))
print(time() - start)


# dataset no. 4 ----------

digits = load_digits()
Z = digits.data
t = digits.target
np.random.seed(123)
Z_train, Z_test, y_train, y_test = train_test_split(Z, t, test_size=0.2)

# create the model with nnetsauce
fit_obj = ns.Ridge2Classifier(lambda1 = 7.11914091e+04, 
                             lambda2 = 4.63867241e+04, 
                             n_hidden_features=13, 
                             n_clusters=0, 
                             row_sample = 7.65039063e-01, 
                             dropout = 5.21582031e-01,
                             type_clust = "gmm")

# fit the model on training set
fit_obj.fit(Z_train, y_train)

# get the accuracy on test set
print(fit_obj.score(Z_test, y_test))
