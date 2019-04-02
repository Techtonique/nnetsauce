#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 19:05:12 2019

@author: moudiki
"""

import numpy as np
from sklearn import datasets, gaussian_process
import nnetsauce as ns
from sklearn.ensemble import VotingClassifier

def bag_custom(B = 10, dropout = 0.5):

    regr = gaussian_process.GaussianProcessClassifier()
        
    estimators = [(str(i), ns.Custom(obj=regr,
                       n_hidden_features=25, dropout=dropout,
                       direct_link=True, bias=True,
                       nodes_sim=np.random.choice(['uniform', 'sobol', 'hammersley']), 
                       activation_name=np.random.choice(['relu', 'sigmoid', 'tanh']),
                       n_clusters=3, seed = i*100))
            for i in range(B)]

    return estimators    


estimators = bag_custom(B = 100, dropout = 0.1)

eclf1 = VotingClassifier(estimators)

breast_cancer = datasets.load_breast_cancer()
X = breast_cancer.data
y = breast_cancer.target

index_train = range(455)
index_test = range(455, 569)
X_train = X[index_train, :]
y_train = y[index_train]
X_test = X[index_test, :]
y_test = y[index_test]

eclf1.fit(X_train, y_train)

# Ensemble score 
print(eclf1.score(X_test, y_test))

# Individual scores
[estimators[i][1].fit(X_train, y_train).score(X_test, y_test) for i in range(100)]
