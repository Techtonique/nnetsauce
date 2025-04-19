import os 
import nnetsauce as ns
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn import datasets, metrics
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.base.datetools import dates_from_str
from time import time

#print(f"\n ----- Running: {os.path.basename(__file__)}... ----- \n")

# Example 1 

url = "https://github.com/ritvikmath/Time-Series-Analysis/raw/master/ice_cream_vs_heater.csv"

df = pd.read_csv(url)

#df.set_index('date', inplace=True)
df.set_index('Month', inplace=True)
df.index.rename('date')
df = df.diff().dropna()

print(df.shape)
print(198*0.8)
# df_train = df.iloc[0:97,]
# df_test = df.iloc[97:123,]
df_train = df.iloc[0:158,]
df_test = df.iloc[158:198,]
h = df_test.shape[0]

print(df_train.head())
print(df_train.tail())
print(df_test.head())
print(df_test.tail())


# Adjust Ridge
regr = linear_model.RidgeCV(alphas=10**np.linspace(-10, 10, 20))
obj_MTS = ns.MTS(regr, 
                 n_hidden_features=5, 
                 verbose = 1, 
                 type_pi="scp2-kde", 
                 replications=250,
                 lags="AIC")
start = time()                 
obj_MTS.fit(df_train)
print("Elapsed:", time()-start)

# with Gaussian prediction intervals
res1 = obj_MTS.predict(h=h, level=95) 
print(res1)
print("\n")

obj_MTS.plot("heater")
obj_MTS.plot("ice cream")

# Example 2

# Adjust Ridge
regr = linear_model.RidgeCV(alphas=10**np.linspace(-10, 10, 20))
obj_MTS = ns.MTS(regr, 
                 n_hidden_features=5, 
                 verbose = 1, 
                 type_pi="gaussian", 
                 lags="AICc")
start = time()                                  
obj_MTS.fit(df_train)
print("Elapsed:", time()-start)

# with Gaussian prediction intervals
res1 = obj_MTS.predict(h=h, level=95) 
print(res1)
print("\n")

obj_MTS.plot("heater")
obj_MTS.plot("ice cream")


# Example 3

# Adjust Ridge
regr = linear_model.RidgeCV(alphas=10**np.linspace(-10, 10, 20))
obj_MTS = ns.MTS(regr, 
                 n_hidden_features=5, 
                 verbose = 1, 
                 type_pi="gaussian", 
                 lags="BIC")
start = time()                                  
obj_MTS.fit(df_train)
print("Elapsed:", time()-start)

# with Gaussian prediction intervals
res1 = obj_MTS.predict(h=h, level=95) 
print(res1)
print("\n")

obj_MTS.plot("heater")
obj_MTS.plot("ice cream")


url = "/Users/t/Documents/datasets/time_series/univariate/AirPassengers.csv"
h = 20

df = pd.read_csv(url)
df.index = pd.DatetimeIndex(df.date) # must have
df.drop(columns=['date'], inplace=True)

# Example 4
# Adjust Ridge
regr = linear_model.RidgeCV(alphas=10**np.linspace(-10, 10, 20))
obj_MTS = ns.MTS(regr, 
                 n_hidden_features=5, 
                 verbose = 1, 
                 replications=250,
                 type_pi="scp2-kde", 
                 lags="AIC")
start = time()                 
obj_MTS.fit(df)
print("Elapsed:", time()-start)

# with Gaussian prediction intervals
res1 = obj_MTS.predict(h=h, level=95) 
print(res1)
print("\n")

obj_MTS.plot()

# Example 5
# Adjust Ridge
regr = linear_model.RidgeCV(alphas=10**np.linspace(-10, 10, 20))
obj_MTS = ns.MTS(regr, 
                 n_hidden_features=5, 
                 verbose = 1, 
                 type_pi="gaussian", 
                 lags="AICc")
start = time()
obj_MTS.fit(df)
print("Elapsed:", time()-start)
# with Gaussian prediction intervals
res1 = obj_MTS.predict(h=h, level=95)
print(res1)
print("\n")
obj_MTS.plot()
# Example 6
# Adjust Ridge
regr = linear_model.RidgeCV(alphas=10**np.linspace(-10, 10, 20))
obj_MTS = ns.MTS(regr, 
                 n_hidden_features=5, 
                 verbose = 1, 
                 type_pi="gaussian", 
                 lags="BIC")
start = time()
obj_MTS.fit(df)
print("Elapsed:", time()-start)
# with Gaussian prediction intervals
res1 = obj_MTS.predict(h=h, level=95)
print(res1)
print("\n")
obj_MTS.plot()
