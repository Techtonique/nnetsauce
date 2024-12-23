import nnetsauce as ns
from sklearn.linear_model import RidgeClassifierCV, LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

X, y = load_iris(return_X_y=True)

model = ns.CustomClassifier(RidgeClassifierCV())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\n\n Objective: abs -----")

res = model.cross_val_score(X_train, y_train, 
                      X_test=X_test, y_test=y_test,
                      scoring='accuracy')

print(res)

model.set_params(n_hidden_features=10)

res = model.cross_val_score(X_train, y_train, 
                      X_test=X_test, y_test=y_test,
                      scoring='accuracy')

print(res)

model.set_params(n_hidden_features=10, dropout=0.5)

res = model.cross_val_score(X_train, y_train, 
                      X_test=X_test, y_test=y_test,
                      scoring='accuracy')

print(res)

model.set_params(n_hidden_features=7, dropout=0.5)

res = model.cross_val_score(X_train, y_train, 
                      X_test=X_test, y_test=y_test,
                      scoring='accuracy')

print(res)

model = ns.CustomClassifier(LogisticRegressionCV())

res = model.cross_val_score(X_train, y_train, 
                      X_test=X_test, y_test=y_test,
                      scoring='accuracy',
                      objective='abs')

print(res)

model.set_params(n_hidden_features=10)

res = model.cross_val_score(X_train, y_train, 
                      X_test=X_test, y_test=y_test,
                      scoring='accuracy',
                      objective='abs')

print(res)

model.set_params(n_hidden_features=10, dropout=0.5)

res = model.cross_val_score(X_train, y_train, 
                      X_test=X_test, y_test=y_test,
                      scoring='accuracy',
                      objective='abs')

print(res)

model.set_params(n_hidden_features=7, dropout=0.5)

res = model.cross_val_score(X_train, y_train, 
                      X_test=X_test, y_test=y_test,
                      scoring='accuracy',
                      objective='abs')

print(res)

print("\n\n Objective: relative -----")

model = ns.CustomClassifier(LogisticRegressionCV())

res = model.cross_val_score(X_train, y_train, 
                      X_test=X_test, y_test=y_test,
                      scoring='accuracy',
                      objective='relative')

print(res)

model.set_params(n_hidden_features=10)

res = model.cross_val_score(X_train, y_train, 
                      X_test=X_test, y_test=y_test,
                      scoring='accuracy',
                      objective='relative')

print(res)

model.set_params(n_hidden_features=10, dropout=0.5)

res = model.cross_val_score(X_train, y_train, 
                      X_test=X_test, y_test=y_test,
                      scoring='accuracy',
                      objective='relative')

print(res)

model.set_params(n_hidden_features=7, dropout=0.5)

res = model.cross_val_score(X_train, y_train, 
                      X_test=X_test, y_test=y_test,
                      scoring='accuracy',
                      objective='relative')

print(res)
