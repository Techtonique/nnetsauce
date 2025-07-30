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
    print("regr.beta_", regr.beta_)

    regr = ns.ElasticNet2Regressor(solver="cd", type_loss='quantile')

    start = time()
    regr.fit(X_train, y_train)
    preds = regr.predict(X_test)
    print("Elapsed: ", time()-start)

    print(f"RMSE for {datasets_names[i]} : {root_mean_squared_error(preds, y_test)}")
    print("regr.beta_", regr.beta_)

    # Example 1: Adam optimizer (Optax)
    regr = ns.ElasticNet2Regressor(
        solver="adam",          # Optax optimizer name
        learning_rate=0.01,     # Learning rate
        max_iter=1000,          # Max iterations
        tol=1e-4,              # Tolerance for early stopping
        verbose=True           # Print progress
    )
    start = time()
    regr.fit(X_train, y_train)
    preds = regr.predict(X_test)
    print(f"Adam - RMSE for {datasets_names[i]}: {root_mean_squared_error(preds, y_test)}")
    print(f"Elapsed: {time() - start:.2f}s\n")
    print("regr.beta_", regr.beta_)

    # Example 2: SGD with momentum (Optax)
    regr = ns.ElasticNet2Regressor(
        solver="sgd",           # Stochastic Gradient Descent
        learning_rate=0.001,    # Smaller learning rate for SGD
        max_iter=1500,
        type_loss='quantile',   # Quantile regression
        quantile=0.5           # Median regression
    )
    start = time()
    regr.fit(X_train, y_train)
    preds = regr.predict(X_test)
    print(f"SGD (Quantile) - RMSE for {datasets_names[i]}: {root_mean_squared_error(preds, y_test)}")
    print(f"Elapsed: {time() - start:.2f}s\n")   
    print("regr.beta_", regr.beta_) 

