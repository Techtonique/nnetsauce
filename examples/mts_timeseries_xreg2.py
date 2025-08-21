import os 
import nnetsauce as ns
import numpy as np
import pandas as pd
from sklearn import datasets, metrics
from sklearn import linear_model
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor

print(f"\n ----- Running: {os.path.basename(__file__)}... ----- \n")

np.random.seed(1235)

url = "https://github.com/ritvikmath/Time-Series-Analysis/raw/master/ice_cream_vs_heater.csv"

df = pd.read_csv(url)

#df.set_index('date', inplace=True)
df.set_index('Month', inplace=True)
df.index.rename('date')

print(df.shape)
print(198*0.8)
# df_train = df.iloc[0:97,]
# df_test = df.iloc[97:123,]
df_train = df.iloc[0:158,]
df_test = df.iloc[158:198,]

print(df_train.head())
print(df_train.tail())
print(df_test.head())
print(df_test.tail())

regr4 = KNeighborsRegressor()
obj_MTS = ns.MTS(ns.QuantileRegressor(regr4), 
                 lags = 1, n_hidden_features=5, verbose = 1)
obj_MTS.fit(df_train)

# with credible intervals
res1 = obj_MTS.predict(return_pi=True) 
print(res1)
print("\n")

# Adjust Bayesian Ridge
regr4 = KNeighborsRegressor()
obj_MTS = ns.MTS(regr4,
                 type_pi="quantile", 
                 level=50, 
                 lags = 5, n_hidden_features=5, verbose = 1)
obj_MTS.fit(df_train)
# with credible intervals
res1 = obj_MTS.predict(h=5) 
print(res1)
print("\n")

obj_MTS = ns.MTS(regr4,
                 type_pi="quantile", 
                 level=95, 
                 lags = 5, n_hidden_features=5, verbose = 1)
obj_MTS.fit(df_train)
# with credible intervals
res1 = obj_MTS.predict(h=5) 
print(res1)
print("\n")

obj_MTS = ns.MTS(regr4,
                 type_pi="quantile", 
                 level=5, 
                 lags = 5, n_hidden_features=5, verbose = 1)
obj_MTS.fit(df_train)
# with credible intervals
res1 = obj_MTS.predict(h=5) 
print(res1)
print("\n")

