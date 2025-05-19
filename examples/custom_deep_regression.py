import matplotlib.pyplot as plt
import os 
import nnetsauce as ns 
import numpy as np
import sklearn.metrics as skm2
import warnings
from sklearn.datasets import load_diabetes, fetch_california_housing
from sklearn.linear_model import LassoCV, RidgeCV, ElasticNetCV, HuberRegressor
from sklearn.model_selection import train_test_split
from time import time 

print(f"\n ----- Running: {os.path.basename(__file__)}... ----- \n")

load_models = [LassoCV, RidgeCV, ElasticNetCV, HuberRegressor]
load_datasets = [load_diabetes(), fetch_california_housing()]

warnings.filterwarnings('ignore')

split_color = 'green'
split_color2 = 'orange'
local_color = 'gray'

def plot_func(x,
              y,
              y_u=None,
              y_l=None,
              pred=None,
              shade_color="lightblue",
              method_name="",
              title=""):

    fig = plt.figure()

    plt.plot(x, y, 'k.', alpha=.3, markersize=10,
             fillstyle='full', label=u'Test set observations')

    if (y_u is not None) and (y_l is not None):
        plt.fill(np.concatenate([x, x[::-1]]),
                 np.concatenate([y_u, y_l[::-1]]),
                 alpha=.3, fc=shade_color, ec='None',
                 label = method_name + ' Prediction interval')

    if pred is not None:
        plt.plot(x, pred, 'k--', lw=2, alpha=0.9,
                 label=u'Predicted value')

    #plt.ylim([-2.5, 7])
    plt.xlabel('$X$')
    plt.ylabel('$Y$')
    plt.legend(loc='upper right')
    plt.title(title)

    plt.show()


for data in load_datasets:
    
    X = data.data
    y= data.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .2, random_state = 13)

    print("\n\n 1 - not conformal \n\n")

    for model in load_models:

        obj = model()

        regr = ns.DeepRegressor(obj, n_layers=3, verbose=1, n_clusters=3, n_hidden_features=2)

        start = time()
        regr.fit(X_train, y_train)
        print(f"\nElapsed: {time() - start} seconds\n")

        print(f"{type(obj).__name__} test RMSE: {regr.score(X_test, y_test)} \n")

    print("\n\n 2 - conformal \n\n")

    print("\n\n 2 - 1 split conformal \n\n")

    for model in load_models:

        obj = model()

        regr = ns.DeepRegressor(obj, n_layers=3, 
                                verbose=1, n_clusters=2, 
                                n_hidden_features=5, 
                                )

        start = time()
        regr.fit(X_train, y_train)
        print(f"\nElapsed: {time() - start} seconds\n")



        preds = regr.predict(X_test, return_pi=True, level=95, 
                             method="splitconformal")
        #print(f"preds: {preds}")
        coverage = np.mean((y_test >= preds.lower) & (y_test <= preds.upper))
        print(f"test coverage: {coverage} \n")
        plot_func(range(len(y_test))[0:30], y_test[0:30],
              preds.upper[0:30], preds.lower[0:30],
              preds.mean[0:30], method_name="Split Conformal")
        # prediction interval average width
        width = np.mean(preds.upper - preds.lower)
        print(f"prediction interval average width: {width} \n")

    print("\n\n 2 - 2 local conformal \n\n")

    for model in load_models:

        obj = model()

        regr = ns.DeepRegressor(obj, n_layers=3, 
                                verbose=1, n_clusters=2, 
                                n_hidden_features=5, 
                                )

        start = time()
        regr.fit(X_train, y_train)
        print(f"\nElapsed: {time() - start} seconds\n")

        preds = regr.predict(X_test, return_pi=True, level=95, 
                             method="localconformal")
        #print(f"preds: {preds}")
        coverage = np.mean((y_test >= preds.lower) & (y_test <= preds.upper))
        print(f"test coverage: {coverage} \n")
