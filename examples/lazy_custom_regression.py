import os 
import nnetsauce as ns 
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

print(f"\n ----- Running: {os.path.basename(__file__)}... ----- \n")

data = load_diabetes()
X = data.data
y= data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .2, random_state = 123)

regr = ns.LazyRegressor(verbose=0, ignore_warnings=True, custom_metric=None, 
                        preprocess=True, 
                        estimators=["RandomForestRegressor", "ExtraTreesRegressor"])
models, predictions = regr.fit(X_train, X_test, y_train, y_test)
model_dictionary = regr.provide_models(X_train, X_test, y_train, y_test)
print(models)
print(model_dictionary["CustomRegressor(RandomForestRegressor)"])

regr = ns.LazyRegressor(verbose=0, ignore_warnings=True, custom_metric=None, 
                        preprocess=True)
models, predictions = regr.fit(X_train, X_test, y_train, y_test)
model_dictionary = regr.provide_models(X_train, X_test, y_train, y_test)
print(models)
print(model_dictionary["CustomRegressor(RandomForestRegressor)"])

regr2 = ns.LazyRegressor(verbose=0, ignore_warnings=True, custom_metric=None, 
                        preprocess=False)
models, predictions = regr2.fit(X_train, X_test, y_train, y_test)
model_dictionary = regr2.provide_models(X_train, X_test, y_train, y_test)
print(models)
print(model_dictionary["CustomRegressor(RandomForestRegressor)"])