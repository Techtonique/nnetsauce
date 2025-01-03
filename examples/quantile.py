import nnetsauce as ns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, Lasso, ElasticNet, RidgeCV, LassoCV, ElasticNetCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.datasets import load_diabetes, fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from tqdm import tqdm


scoring = ["conformal", "residuals", "predictions", "studentized", "conformal-studentized"]

datasets = [load_diabetes, fetch_california_housing]

dataset_names = ["diabetes", "california_housing"]

regrs = [MLPRegressor(), RandomForestRegressor(), RidgeCV(), KNeighborsRegressor()]

for dataset, dataset_name in zip(datasets, dataset_names):

    print("\n dataset", dataset_name)

    X, y = dataset(return_X_y=True)
    if dataset_name == "california_housing":
        X, y = X[:1000], y[:1000]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                        random_state=42)

    for score in tqdm(scoring):

        print("\n score", score)

        for regr in regrs:

            print("\n regr", regr.__class__.__name__)

            regressor = ns.QuantileRegressor(
                base_regressor=regr, 
                score = score   
            )

            regressor.fit(X_train, y_train)
            predictions = regressor.predict(X_test)

            # Check ordering
            lower_bound, median, upper_bound = predictions.lower, predictions.median, predictions.upper
            is_ordered = np.all(np.logical_and(lower_bound < median, median < upper_bound))
            print(f"Are the predictions ordered correctly? {is_ordered}")

            # Calculate coverage
            coverage = np.mean((lower_bound <= y_test)*(upper_bound >= y_test))
            print(f"Coverage: {coverage:.2f}")

            # Plot
            plt.figure(figsize=(10, 6))
            
            # Plot the actual values
            plt.plot(y_test, label='Actual', color='black', alpha=0.5)
            
            # Plot the predictions and confidence interval
            plt.plot(predictions.median, label='Median prediction', color='blue', linewidth=2)
            plt.plot(predictions.mean, label='Mean prediction', color='orange', linestyle='--', linewidth=2)
            plt.fill_between(range(len(y_test)), 
                           lower_bound, upper_bound,
                           alpha=0.3, color='gray',
                           edgecolor='gray',
                           label='Prediction interval')
            
            plt.title(f'{regr.__class__.__name__} - {score} scoring')
            plt.xlabel('Sample index')
            plt.ylabel('Value')
            plt.legend()
            plt.show()