import copy
import nnetsauce as ns 
import numpy as np
from sklearn.datasets import load_diabetes, fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, RidgeCV, LassoCV, SGDRegressor


def check_is_fitted(estimator):
    return hasattr(estimator, "coef_")


for alpha in [0.1, 0.3, 0.5, 0.7, 0.9]:

    print(f"\n\n alpha: {alpha} ---------------\n")

    # load data
    for dataset in [load_diabetes, fetch_california_housing]:

        print(f"\n\n{dataset.__name__} ----------\n")

        for model in [LinearRegression, Ridge, Lasso, RidgeCV, LassoCV]: 

            print(f"\n{model.__name__} -----\n")   

            X, y = dataset(return_X_y=True)

            # split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

            # fit a linear regression model
            regr = model().fit(X_train, y_train)

            # update the linear regression model
            regr_upd = ns.RegressorUpdater(regr, alpha=alpha)
            regr_upd.fit(X_train, y_train)
            print(f"coef: {regr_upd.coef_}")
            regr_upd.partial_fit(X_test[0, :], y_test[0])
            print(f"RMSE: { np.sqrt(np.mean((regr_upd.predict(X_test[1:]) - y_test[1:])**2)) }")
            print(f"coef: {regr_upd.coef_}")
            regr_upd.partial_fit(X_test[1, :], y_test[1])
            print(f"RMSE: { np.sqrt(np.mean((regr_upd.predict(X_test[2:]) - y_test[2:])**2)) }") 
            print(f"coef: {regr_upd.coef_}")
            regr_upd.partial_fit(X_test[2, :], y_test[2])
            print(f"RMSE: { np.sqrt(np.mean((regr_upd.predict(X_test[3:]) - y_test[3:])**2)) }")
            print(f"coef: {regr_upd.coef_}")
            regr_upd.partial_fit(X_test[3, :], y_test[3])
            print(f"RMSE: { np.sqrt(np.mean((regr_upd.predict(X_test[4:]) - y_test[4:])**2)) }")
            print(f"coef: {regr_upd.coef_}")




for alpha in [0.1, 0.3, 0.5, 0.7, 0.9]:

    print(f"\n\n alpha: {alpha} ---------------\n")

    # load data
    for dataset in [load_diabetes, fetch_california_housing]:

        print(f"\n\n{dataset.__name__} ----------\n")

        for model in [SGDRegressor]: #, Ridge, Lasso, RidgeCV, LassoCV]: 

            print(f"\n{model.__name__} -----\n")   

            X, y = dataset(return_X_y=True)

            # split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

            # fit a linear regression model
            regr = ns.CustomRegressor(obj=model()).fit(X_train, y_train)

            # update the linear regression model
            regr_upd = ns.RegressorUpdater(regr, alpha=alpha)
            regr_upd.fit(X_train, y_train)
            print(f"coef: {regr_upd.coef_}")
            regr_upd.partial_fit(X_test[0, :], y_test[0])
            print(f"RMSE: { np.sqrt(np.mean((regr_upd.predict(X_test[1:]) - y_test[1:])**2)) }")
            print(f"coef: {regr_upd.coef_}")
            regr_upd.partial_fit(X_test[1, :], y_test[1])
            print(f"RMSE: { np.sqrt(np.mean((regr_upd.predict(X_test[2:]) - y_test[2:])**2)) }") 
            print(f"coef: {regr_upd.coef_}")
            regr_upd.partial_fit(X_test[2, :], y_test[2])
            print(f"RMSE: { np.sqrt(np.mean((regr_upd.predict(X_test[3:]) - y_test[3:])**2)) }")
            print(f"coef: {regr_upd.coef_}")
            regr_upd.partial_fit(X_test[3, :], y_test[3])
            print(f"RMSE: { np.sqrt(np.mean((regr_upd.predict(X_test[4:]) - y_test[4:])**2)) }")
            print(f"coef: {regr_upd.coef_}")


