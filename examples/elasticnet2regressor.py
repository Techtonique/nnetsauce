import nnetsauce as ns 
import numpy as np 
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes, fetch_california_housing
from time import time 

load_datasets = [load_diabetes(), fetch_california_housing()]

datasets_names = ["diabetes", "housing"]

for i, data in enumerate(load_datasets):
    
    X = data.data
    y= data.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .2, random_state = 13)

    regr = ns.ElasticNet2Regressor(solver="cd")

    start = time()
    regr.fit(X_train, y_train)
    preds = regr.predict(X_test)
    print("Elapsed: ", time()-start)

    print(f"RMSE for {datasets_names[i]} : {root_mean_squared_error(preds, y_test)}")

    regr = ns.ElasticNet2Regressor(solver="cd", type_loss='quantile')

    start = time()
    regr.fit(X_train, y_train)
    preds = regr.predict(X_test)
    print("Elapsed: ", time()-start)

    print(f"RMSE for {datasets_names[i]} : {root_mean_squared_error(preds, y_test)}")

