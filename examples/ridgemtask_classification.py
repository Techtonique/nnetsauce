import os 
import nnetsauce as ns
import numpy as np
from sklearn.datasets import load_breast_cancer, load_wine, load_iris, load_digits, make_classification
from sklearn.model_selection import train_test_split
from sklearn import metrics
from time import time

print(f"\n ----- Running: {os.path.basename(__file__)}... ----- \n")

# dataset no. 1 ---------- 

print(" \n breast cancer dataset ----- \n")

breast_cancer = load_breast_cancer()
Z = breast_cancer.data
t = breast_cancer.target
np.random.seed(123)
X_train, X_test, y_train, y_test = train_test_split(Z, t, test_size=0.2)

fit_obj = ns.Ridge2MultitaskClassifier(n_hidden_features=int(9.83730469e+01), 
                                   dropout=4.31054687e-01, 
                                   n_clusters=int(1.71484375e+00),
                                   lambda1=1.24023438e+01, lambda2=7.30263672e+03)

start = time()
fit_obj.fit(X_train, y_train)
print(f"Elapsed {time() - start}") 

print(fit_obj.score(X_test, y_test))

start = time()
preds = fit_obj.predict(X_test)
print(f"Elapsed {time() - start}") 
print(metrics.classification_report(preds, y_test))


# dataset no. 2 ----------

print(" \n wine dataset ----- \n")

wine = load_wine()
Z = wine.data
t = wine.target
np.random.seed(123)
Z_train, Z_test, y_train, y_test = train_test_split(Z, t, test_size=0.2)

fit_obj = ns.Ridge2MultitaskClassifier(n_hidden_features=15,
                                  dropout=0.1, n_clusters=3, 
                                  type_clust="gmm")

start = time()
fit_obj.fit(Z_train, y_train)
print(f"Elapsed {time() - start}") 

print(fit_obj.score(Z_test, y_test))

preds = fit_obj.predict(Z_test)
print(metrics.classification_report(preds, y_test))

# dataset no. 4 ----------

print(" \n make_classification dataset ----- \n")

X, y = make_classification(n_samples=2500, n_features=20, 
                                               random_state=783451)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                    random_state=351452)


fit_obj = ns.Ridge2MultitaskClassifier(n_hidden_features=5,
                                  dropout=0.1, n_clusters=3, 
                                  type_clust="gmm")

start = time()
fit_obj.fit(X_train, y_train)
print(f"Elapsed {time() - start}") 

print(fit_obj.score(X_test, y_test))

preds = fit_obj.predict(X_test)
print(metrics.classification_report(preds, y_test))

# dataset no. 5 ----------

print(" \n digits dataset ----- \n")

digits = load_digits()
X = digits.data
y = digits.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=123)

fit_obj = ns.Ridge2MultitaskClassifier(n_hidden_features=25,
                                  dropout=0.1, n_clusters=3, 
                                  type_clust="gmm")

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

iris = load_iris()
Z = iris.data
t = iris.target
np.random.seed(123)
Z_train, Z_test, y_train, y_test = train_test_split(Z, t, test_size=0.2,
                                                    random_state=123)

fit_obj = ns.Ridge2MultitaskClassifier(n_hidden_features=10,
                                       dropout=0.1, n_clusters=2)

start = time()
fit_obj.fit(Z_train, y_train)
print(f"Elapsed {time() - start}") 

print(fit_obj.score(Z_test, y_test))


