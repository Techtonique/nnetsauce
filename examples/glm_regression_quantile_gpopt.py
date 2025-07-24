import os 
import GPopt as gp
import nnetsauce as ns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from time import time
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import pinball_loss, make_scorer

# Define the custom scorer using sklearn's pinball_loss
def mean_pinball_loss(y_true, y_pred, quantile=0.5):
    return pinball_loss(y_true, y_pred, alpha=quantile)

# Create the scorer for use in cross_val_score
mean_pinball_scorer = make_scorer(mean_pinball_loss, quantile=0.5, greater_is_better=False)

print(f"\n ----- Running: {os.path.basename(__file__)}... ----- \n")

X, y = fetch_california_housing(return_X_y=True)    
X, y = X[:1000], y[:1000]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=2020)
X_train_val, X_test_val, y_train_val, y_test_val = train_test_split(X_train, y_train, random_state=2020)

def objective(xx):
    learning_rate = 10**xx[0]
    batch_prop = xx[1]
    n_hidden_features = int(10**xx[2])
    lambda1 = 10**xx[3]
    alpha1 = xx[4]
    lambda2 = 10**xx[5]
    alpha2 = xx[6]
    try: 
        opt = ns.Optimizer(type_optim="scd", 
                            learning_rate=learning_rate, 
                            batch_prop=batch_prop, 
                            verbose=0)
        obj = ns.GLMRegressor(n_hidden_features=n_hidden_features, 
                            lambda1=lambda1, alpha1=alpha1,
                            lambda2=lambda2, alpha2=alpha2, 
                            family="quantile",
                            level=97.5,
                            optimizer=opt)    
        # Use cross_val_score to compute the mean pinball score
        scores = cross_val_score(obj, X_train_val, y_train_val, cv=5, scoring=mean_pinball_scorer)
        pred_upper = obj.predict(X_test_val)
        if np.any(np.isnan(pred_upper)):
            return 1e6

        opt = ns.Optimizer(type_optim="scd", 
                            learning_rate=learning_rate, 
                            batch_prop=batch_prop, 
                            verbose=0)
        obj = ns.GLMRegressor(n_hidden_features=n_hidden_features, 
                            lambda1=lambda1, alpha1=alpha1,
                            lambda2=lambda2, alpha2=alpha2, 
                            family="quantile",
                            level=2.5,
                            optimizer=opt)    
        obj.fit(X_train_val, y_train_val)
        pred_lower = obj.predict(X_test_val)
        if np.any(np.isnan(pred_lower)):
            return 1e6

        length = pred_upper - pred_lower
        penalty1 = np.maximum(pred_lower - y_test_val, 0)
        penalty2 = np.maximum(y_test_val - pred_lower, 0)
        #res = np.mean(length + (2/0.2)*(penalty1 + penalty2))
        res = -np.mean((pred_upper >= y_test_val)*(pred_lower <= y_test_val)) #+ np.mean(length + (2/0.05)*(penalty1 + penalty2))
        return res 
    except Exception as e:
        print(f"Error: {e}") 
        return 1e6


#learning_rate = 10**xx[0]
#batch_prop = xx[1]
#n_hidden_features = xx[2]
#lambda1 = 10**xx[3]
#alpha1 = xx[4]
#lambda2 = 10**xx[5]
#alpha2 = xx[6]
gp_opt = gp.GPOpt(objective_func=objective,
                          lower_bound = np.array([-4, 0.0, 0.5,  -4, 0, -4, 0]),
                          upper_bound = np.array([-1, 0.8, 2.5, 5, 1, 5, 1]),
                          params_names=["learning_rate", "batch_prop", "n_hidden_features", 
                          "lambda1", "alpha1", "lambda2", "alpha2"],
                          n_init=10,
                          n_iter=190,
                          seed=432)
res = gp_opt.optimize(verbose=2)
print(res)

#learning_rate = 10**xx[0]
#batch_prop = xx[1]
#n_hidden_features = xx[2]
#lambda1 = 10**xx[3]
#alpha1 = xx[4]
#lambda2 = 10**xx[5]
#alpha2 = xx[6]
res.best_params["learning_rate"] = 10**res.best_params["learning_rate"]
res.best_params["n_hidden_features"] = int(10**res.best_params["n_hidden_features"])
res.best_params["lambda1"] = 10**res.best_params["lambda1"]
res.best_params["lambda2"] = 10**res.best_params["lambda2"]



opt = ns.Optimizer(type_optim="scd", 
                    learning_rate=res.best_params["learning_rate"], 
                    batch_prop=res.best_params["batch_prop"], 
                    verbose=0)
obj = ns.GLMRegressor(n_hidden_features=res.best_params["n_hidden_features"], 
                      lambda1=res.best_params["lambda1"], alpha1=res.best_params["alpha1"],
                     lambda2=res.best_params["lambda2"], alpha2=res.best_params["alpha2"], 
                    family="quantile",
                    level=97.5,
                    optimizer=opt)    
obj.fit(X_train, y_train)
pred_upper = obj.predict(X_test)

opt = ns.Optimizer(type_optim="scd", 
                    learning_rate=res.best_params["learning_rate"], 
                    batch_prop=res.best_params["batch_prop"], 
                    verbose=0)
obj = ns.GLMRegressor(n_hidden_features=res.best_params["n_hidden_features"], 
                      lambda1=res.best_params["lambda1"], alpha1=res.best_params["alpha1"],
                     lambda2=res.best_params["lambda2"], alpha2=res.best_params["alpha2"], 
                    family="quantile",
                    level=2.5,
                    optimizer=opt)    
obj.fit(X_train, y_train)
pred_lower = obj.predict(X_test)
print(np.mean((pred_upper >= y_test)*(pred_lower <= y_test)))
print(pred_upper)
print(pred_lower)
print(y_test)