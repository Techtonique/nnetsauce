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

print(f"\n ----- Running: {os.path.basename(__file__)}... ----- \n")

subprocess.check_call([sys.executable, "-m", "pip", "install", "statsmodels"])
from statsmodels.stats.diagnostic import acorr_ljungbox

np.random.seed(1235)

# url = "https://raw.githubusercontent.com/selva86/datasets/master/Raotbl6.csv"
# url = "/Users/t/Documents/datasets/time_series/multivariate/Raotbl6.csv"
url = "https://github.com/ritvikmath/Time-Series-Analysis/raw/master/ice_cream_vs_heater.csv"

df = pd.read_csv(url)

#df.set_index('date', inplace=True)
df.set_index('Month', inplace=True)
df.index.rename('date')

print(df.shape)
print(198*0.8)
df_train = df.iloc[0:158,]
df_test = df.iloc[158:198,]

print(f"df_train.head():\n{df_train.head()}")
print(f"df_train.tail():\n{df_train.tail()}")
print(f"df_test.head():\n{df_test.head()}")

print("\n\n Example 1: MTS with linear_model.BayesianRidge() ----------------- \n")

regr = linear_model.BayesianRidge()
obj_MTS = ns.MTS(regr, lags = 1, n_hidden_features=5, verbose = 1)
obj_MTS.fit(df_train, xreg=np.arange(df_train.shape[0]))
print("\n")
print(obj_MTS.predict(h=5, return_std=True))

print("\n\n Example 2: MTS with GaussianProcessRegressor() -----------------")

regr = GaussianProcessRegressor()
obj_MTS = ns.MTS(regr, lags = 1, n_hidden_features=5, verbose = 1)
obj_MTS.fit(df_train.values, xreg=np.arange(df_train.shape[0]))
#obj_MTS.fit(df_train.values, xreg=np.flip(np.arange(df_train.shape[0])))
print("\n")
print(obj_MTS.predict(h=5, return_std=True))

print("\n\n Example 3: MTS with linear_model.ARDRegression() -----------------")

regr = linear_model.ARDRegression()
obj_MTS = ns.MTS(regr, lags = 1, n_hidden_features=5, verbose = 1)
obj_MTS.fit(df_train.values, xreg=np.arange(df_train.shape[0]))
#obj_MTS.fit(df_train.values, xreg=np.flip(np.arange(df_train.shape[0])))
print("\n")
print(obj_MTS.predict(h=5, return_std=True))

print("\n\n Example 4: MTS with BayesianRidge() -----------------")

regr = linear_model.BayesianRidge()
obj_MTS = ns.MTS(regr, lags = 1, n_hidden_features=5, verbose = 1)
obj_MTS.fit(df_train.iloc[:,0].values, xreg=np.arange(df_train.shape[0]))
#obj_MTS.fit(df_train.values, xreg=np.flip(np.arange(df_train.shape[0])))
print("\n")
print(obj_MTS.predict(h=5, return_std=True))

print("\n\n Example 5: MTS with BayesianRidge() -----------------")

regr = linear_model.BayesianRidge()
obj_MTS = ns.MTS(regr, lags = 1, n_hidden_features=5, verbose = 1)
obj_MTS.fit(df_train.iloc[:,0].values)
#obj_MTS.fit(df_train.values, xreg=np.flip(np.arange(df_train.shape[0])))
print("\n")
print(obj_MTS.predict(h=5, return_std=True))

print("\n\n Example 6: MTS with BayesianRidge() -----------------")

df_train["heater"]
regr = linear_model.BayesianRidge()
obj_MTS = ns.MTS(regr, lags = 1, n_hidden_features=5, verbose = 1)
obj_MTS.fit(df_train, 
            xreg=np.column_stack((np.arange(df_train.shape[0]), np.random.randn(df_train.shape[0]))))
#obj_MTS.fit(df_train.values, xreg=np.flip(np.arange(df_train.shape[0])))
print("\n")
print(obj_MTS.predict(h=5, return_std=True))


print("\n\n Example 7: MTS with BayesianRidge() -----------------")

regr = linear_model.BayesianRidge()
obj_MTS = ns.MTS(regr, lags = 2, n_hidden_features=5, verbose = 1)
obj_MTS.fit(df_train, 
            xreg=np.column_stack((np.arange(df_train.shape[0]), np.random.randn(df_train.shape[0]))))
#obj_MTS.fit(df_train.values, xreg=np.flip(np.arange(df_train.shape[0])))
print("\n")
print(obj_MTS.predict(h=5, return_std=True))

