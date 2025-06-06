import os 
import nnetsauce as ns 
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

print(f"\n ----- Running: {os.path.basename(__file__)}... ----- \n")

# without preprocessing
print("\n\nWithout preprocessing")

data = load_diabetes()
X = data.data
y= data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .2, 
                                                    random_state = 123)

regr2 = ns.LazyDeepRegressor(n_layers=3, n_hidden_features=2, 
verbose=0, ignore_warnings=True, estimators=["ExtraTreesRegressor", 
                                             "RandomForestRegressor", 
                                             "LassoLarsIC"])
models = regr2.fit(X_train, X_test, y_train, y_test)
model_dictionary = regr2.provide_models(X_train, X_test, y_train, y_test)
print(models)


regr = ns.LazyDeepRegressor(n_layers=3, n_hidden_features=2, 
verbose=0, ignore_warnings=True)
models = regr.fit(X_train, X_test, y_train, y_test)
model_dictionary = regr.provide_models(X_train, X_test, y_train, y_test)
print(models)


# with preprocessing
print("\n\nWith preprocessing")

data = load_diabetes()
X = data.data
y= data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .2, 
                                                    random_state = 123)

regr2 = ns.LazyDeepRegressor(n_layers=3, n_hidden_features=2, 
verbose=0, ignore_warnings=True, estimators=["ExtraTreesRegressor", 
                                             "RandomForestRegressor", 
                                             "LassoLarsIC"], 
                                             preprocess=True)
models = regr2.fit(X_train, X_test, y_train, y_test)
model_dictionary = regr2.provide_models(X_train, X_test, y_train, y_test)
print(models)


regr = ns.LazyDeepRegressor(n_layers=3, n_hidden_features=2, 
verbose=0, ignore_warnings=True, preprocess=True)
models = regr.fit(X_train, X_test, y_train, y_test)
model_dictionary = regr.provide_models(X_train, X_test, y_train, y_test)
print(models)

