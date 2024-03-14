import os
import subprocess
import sys 
import nnetsauce as ns
import numpy as np
import pandas as pd
from scipy import stats
from sklearn import datasets, metrics
from sklearn import linear_model
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.ensemble import RandomForestRegressor
from time import time
from sklearn.gaussian_process.kernels import Matern

print(f"\n ----- Running: {os.path.basename(__file__)}... ----- \n")

subprocess.check_call([sys.executable, "-m", "pip", "install", "statsmodels"])
from statsmodels.stats.diagnostic import acorr_ljungbox

np.random.seed(1235)

url = "https://github.com/ritvikmath/Time-Series-Analysis/raw/master/ice_cream_vs_heater.csv"

df = pd.read_csv(url)

#df.set_index('date', inplace=True)
df.set_index('Month', inplace=True)
df.index.rename('date')

print(df.shape)
print(198*0.8)
df_train = df.iloc[0:150,]
df_test = df.iloc[150:155,]


print(f"df_train.head():\n{df_train.head()}")
print(f"df_train.tail():\n{df_train.tail()}")
print(f"df_test.head():\n{df_test.head()}")


regr = GaussianProcessRegressor(kernel=Matern(nu=1.5),
                                alpha=1e-6,
                                normalize_y=True,
                                n_restarts_optimizer=25,
                                random_state=42,)

print("\n\n Example: MTS2 with GaussianProcessRegressor() s -----------------")

obj_MTS2 = ns.MTS(regr, lags = 1, n_hidden_features=5, 
                  replications=4, kernel='gaussian', 
                  verbose = 1, type_pi="kde")
obj_MTS2.fit(df_train.values)
preds2 = obj_MTS2.predict(h=5)
print(preds2)
print(f" Predictive simulations #1 {obj_MTS2.sims_[0]}") 
print(f" Predictive simulations #2 {obj_MTS2.sims_[1]}") 
print(f" Predictive simulations #3 {obj_MTS2.sims_[2]}") 

print("\n\n Example: MTS with GaussianProcessRegressor() sampling -----------------")

obj_MTS = ns.MTS(regr, lags = 1, n_hidden_features=5, verbose = 1,
                 replications=4)
obj_MTS.fit(df_train.values)

print("\n")
preds = obj_MTS.predict(h=5, sampling=True)
print(len(preds))
print(preds)
print(f" Predictive simulations #1 {obj_MTS.sims_[0]}") 
print(f" Predictive simulations #2 {obj_MTS.sims_[1]}") 
print(f" Predictive simulations #3 {obj_MTS.sims_[2]}") 

