import os
import nnetsauce as ns
import numpy as np
from sklearn.datasets import load_breast_cancer, load_wine, load_iris, make_classification
from sklearn.model_selection import train_test_split
from sklearn import metrics
from time import time
import matplotlib.pyplot as plt

print(f"\n ----- Running: {os.path.basename(__file__)}... ----- \n")

print(f"\n method = 'momentum' ----------")


# dataset no. 1 ---------- 

breast_cancer = load_breast_cancer()
Z = breast_cancer.data
t = breast_cancer.target
np.random.seed(123)
X_train, X_test, y_train, y_test = train_test_split(Z, t, test_size=0.2)

print(f"\n 1 - breast_cancer dataset ----------")
fit_obj = ns.GLMClassifier(n_hidden_features=5, 
                           n_clusters=2, type_clust="gmm")

start = time()
fit_obj.fit(X_train, y_train, verbose=0)
print(time() - start)

# plt.plot(fit_obj.optimizer.results[2])

print(fit_obj.score(X_test, y_test))

start = time()
preds = fit_obj.predict(X_test)
print(time() - start)
print(metrics.classification_report(preds, y_test))


# dataset no. 2 ---------- 

wine = load_wine()
Z = wine.data
t = wine.target
np.random.seed(123575)
X_train, X_test, y_train, y_test = train_test_split(Z, t, test_size=0.2)

print(f"\n 2 - wine dataset ----------")
fit_obj = ns.GLMClassifier(n_hidden_features=3, 
                           n_clusters=2, type_clust="gmm")

start = time()
fit_obj.fit(X_train, y_train, verbose=0)
print(time() - start)

# plt.plot(fit_obj.optimizer.results[2])

print(fit_obj.score(X_test, y_test))

start = time()
preds = fit_obj.predict(X_test)
print(time() - start)
print(metrics.classification_report(preds, y_test))

print(f"\n method = 'exp' ----------")

# dataset no. 1 ---------- 

breast_cancer = load_breast_cancer()
Z = breast_cancer.data
t = breast_cancer.target
X_train, X_test, y_train, y_test = train_test_split(Z, t, test_size=0.2, random_state=123)

print(f"\n 1 - breast_cancer dataset ----------")
opt = ns.Optimizer()
opt.learning_method = "exp"
fit_obj = ns.GLMClassifier(optimizer=opt)
fit_obj.lambda1=1e-5 
fit_obj.lambda2=100
fit_obj.optimizer.type_optim = "scd"
fit_obj.optimizer.verbose=0
fit_obj.optimizer.learning_rate=0.01 
fit_obj.optimizer.batch_prop=0.5

start = time()
fit_obj.fit(X_train, y_train)
print(time() - start)

# plt.plot(fit_obj.optimizer.results[2])

print(fit_obj.score(X_test, y_test))
print(fit_obj.score(X_test, y_test, scoring="roc_auc"))

start = time()
preds = fit_obj.predict(X_test)
print(time() - start)
print(metrics.classification_report(preds, y_test))


print(f"\n method = 'poly' ----------")

# dataset no. 1 ---------- 

print(f"\n 1 - breast_cancer dataset ----------")
opt = ns.Optimizer()
opt.learning_method = "poly"
fit_obj = ns.GLMClassifier(optimizer=opt)
fit_obj.lambda1=1 
fit_obj.lambda2=1
fit_obj.optimizer.type_optim = "scd"
fit_obj.optimizer.verbose=0
fit_obj.optimizer.learning_rate=0.001 
fit_obj.optimizer.batch_prop=0.5

start = time()
fit_obj.fit(X_train, y_train)
print(time() - start)

# plt.plot(fit_obj.optimizer.results[2])

print(fit_obj.score(X_test, y_test))
print(fit_obj.score(X_test, y_test, scoring="roc_auc"))

start = time()
preds = fit_obj.predict(X_test)
print(time() - start)
print(metrics.classification_report(preds, y_test))



# dataset no. 3 ---------- 

iris = load_iris()
Z = iris.data
t = iris.target
X_train, X_test, y_train, y_test = train_test_split(Z, t, test_size=0.2, stratify=t, random_state=123)

print(f"\n 3 - iris dataset ----------")
fit_obj = ns.GLMClassifier(n_hidden_features=3, 
                           n_clusters=3, type_clust="gmm")

start = time()
fit_obj.fit(X_train, y_train)
print(time() - start)

# plt.plot(fit_obj.optimizer.results[2])

print(fit_obj.score(X_test, y_test))

start = time()
preds = fit_obj.predict(X_test)
print(time() - start)
print(metrics.classification_report(preds, y_test))

# dataset no. 4 ----------

X, y = make_classification(n_samples=2500, n_features=20, 
                                               random_state=783451)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y,
                                                    random_state=35145)

print(f"\n 4 - make_classification dataset ----------")
fit_obj = ns.GLMClassifier(n_hidden_features=5,
                                  dropout=0.1, n_clusters=0)

start = time()
fit_obj.fit(X_train, y_train)
print(time() - start)

print(fit_obj.score(X_test, y_test))

preds = fit_obj.predict(X_test)
print(metrics.classification_report(preds, y_test))