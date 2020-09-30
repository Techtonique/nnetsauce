import nnetsauce as ns
import numpy as np
from sklearn.datasets import load_breast_cancer, load_wine, load_iris, make_classification
from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from time import time


# dataset no. 1 ---------- 

breast_cancer = load_breast_cancer()
Z = breast_cancer.data
t = breast_cancer.target
np.random.seed(123)
X_train, X_test, y_train, y_test = train_test_split(Z, t, test_size=0.2)

# Linear Regression is used 
regr = LinearRegression()
fit_obj = ns.MultitaskClassifier(regr, n_hidden_features=5, 
                             n_clusters=2, type_clust="gmm")

start = time()
fit_obj.fit(X_train, y_train)
print(f"Elapsed {time() - start}") 

print(fit_obj.score(X_test, y_test))
print(fit_obj.score(X_test, y_test, scoring="roc_auc"))

start = time()
preds = fit_obj.predict(X_test)
print(f"Elapsed {time() - start}") 
print(metrics.classification_report(preds, y_test))
