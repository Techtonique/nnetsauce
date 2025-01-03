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
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


scoring = ["conformal", "residuals", "predictions"]

for score in scoring:

    regr = Ridge()

    regressor = ns.QuantileRegressor(
        base_regressor=regr, 
        score = score   
    )

    X, y = load_diabetes(return_X_y=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                        random_state=42)

    regressor.fit(X_train, y_train)
    predictions = regressor.predict(X_test)

    # Check ordering
    lower_bound, median, upper_bound = predictions
    is_ordered = np.all(np.logical_and(lower_bound < median, median < upper_bound))
    print(f"Are the predictions ordered correctly? {is_ordered}")

    df = pd.DataFrame(predictions).T
    df.columns = ['lower_bound', 'median', 'upper_bound']

    print("coverage", np.mean((df['lower_bound'] <= y_test)*(df['upper_bound'] >= y_test)))

    df.plot()
    plt.plot(y_test, label='y_test', color='gray', linestyle='--')
    plt.legend()
    plt.show()