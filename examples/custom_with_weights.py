import nnetsauce as ns
import numpy as np 
import os
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

print(f"\n ----- Running: {os.path.basename(__file__)}... ----- \n")

print("Example 1 - classification")

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=123)

clf = ns.CustomClassifier(obj=LogisticRegression())

n_zeros = np.sum(y_train == 0)
n_ones = np.sum(y_train == 1)
weights = np.where(y_train == 0, 1/n_zeros, 1/n_ones)

clf.fit(X_train, y_train, sample_weight=weights)

print(clf.score(X_test, y_test))

clf.fit(X_train, y_train)

print(clf.score(X_test, y_test))

clf = ns.DeepClassifier(obj=LogisticRegression())

clf.fit(X_train, y_train, sample_weight=weights)

print(clf.score(X_test, y_test))

clf.fit(X_train, y_train)

print(clf.score(X_test, y_test))

print("Example 2 - regression")

X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=123)

reg = ns.CustomRegressor(obj=RandomForestRegressor())

weights = np.random.rand(X_train.shape[0])

reg.fit(X_train, y_train, sample_weight=weights)

print(reg.score(X_test, y_test))

reg.fit(X_train, y_train)

print(reg.score(X_test, y_test))

reg = ns.DeepRegressor(obj=RandomForestRegressor())

reg.fit(X_train, y_train, sample_weight=weights)

print(reg.score(X_test, y_test))

reg.fit(X_train, y_train)

print(reg.score(X_test, y_test))


