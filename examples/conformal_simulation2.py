import os 
import nnetsauce as ns 
import matplotlib.pyplot as plt 
import numpy as np 
import warnings
from sklearn.datasets import fetch_california_housing, load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import BayesianRidge, ARDRegression, RidgeCV
from sklearn.ensemble import ExtraTreesRegressor
from time import time 


# # 2 - Useful plotting functions



warnings.filterwarnings('ignore')

split_color = 'green'
split_color2 = 'tomato'
local_color = 'gray'

def plot_func(x,
              y,
              y_u=None,
              y_l=None,
              pred=None,
              shade_color="",
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


# # 3 - Examples of use



data = fetch_california_housing()
X = data.data
y= data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .2, random_state = 123)


# ## RidgeCV 
"""
- 'bootstrap': Bootstrap resampling.
- 'kde': Kernel Density Estimation.
- 'ecdf': Empirical CDF-based sampling.
- 'permutation': Permutation resampling.
- 'smooth_bootstrap': Smoothed bootstrap with added noise.
"""                      

for type_pi in ('bootstrap', 'kde', 'ecdf', 'permutation', 'smooth_bootstrap'):
    print(f"\n\n### type_pi = {type_pi} ###\n")
    regr1 = ns.PredictionInterval(RidgeCV(),
                            replications=100,
                            type_pi=type_pi) # 5 hidden nodes, ReLU activation function
    regr1.fit(X_train, y_train)
    start = time()
    preds1 = regr1.predict(X_test, return_pi=True)
    print(f"Elapsed: {time() - start}s")
    print(f"coverage_rate conformalized QRNN RidgeCV: {np.mean((preds1[2]<=y_test)*(preds1[3]>=y_test))}")
    print(f"predictive simulations: {preds1[1]}")

    max_idx = 50
    plot_func(x = range(max_idx),
            y = y_test[0:max_idx],
            y_u = preds1.upper[0:max_idx],
            y_l = preds1.lower[0:max_idx],
            pred = preds1.mean[0:max_idx],
            shade_color=split_color2,
            title = f"conformalized QRNN RidgeCV ({max_idx} first points in test set)")


