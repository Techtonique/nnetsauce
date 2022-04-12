import nnetsauce as ns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets, metrics
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
np.random.seed(123)


M = np.random.rand(10, 3)
M[:,0] = 10*M[:,0]
M[:,2] = 25*M[:,2]

print(M)
print("\n")

# Adjust Bayesian Ridge
regr4 = linear_model.BayesianRidge()
obj_MTS = ns.MTS(regr4, lags = 1, n_hidden_features=5)
obj_MTS.fit(M)
print(obj_MTS.df_)
print(obj_MTS.predict(return_std=True))

# with credible intervals
print(obj_MTS.predict(return_std=True, level=80))
print("\n")

print(obj_MTS.predict(return_std=True, level=95))
print("\n")

# example with dataframes (#1)
print("examples with dataframes ----- \n")

print("example 1 with dataframes ----- \n")

dataset = {
    'date' : ['2001-01-01', '2002-01-01', '2003-01-01', '2004-01-01', '2005-01-01'],
    'series1' : [34, 30, 35.6, 33.3, 38.1],
    'series2' : [4, 5.5, 5.6, 6.3, 5.1],
    'series3' : [100, 100.5, 100.6, 100.2, 100.1]}
df = pd.DataFrame(dataset).set_index('date')
print(df.shape)
print(df.values)
print(df)

regr5 = linear_model.BayesianRidge()
obj_MTS = ns.MTS(regr5, lags = 1, n_hidden_features=5)
obj_MTS.fit(df)
print(obj_MTS.predict())
print("\n")
print(obj_MTS.predict(return_std=True))

# example with dataframes (#2)

print("\n")
print("example 2 with dataframes ----- \n")

# Data frame containing the time series
dataset = {
'date' : ['2020-01-01', '2020-01-02', '2020-01-03', '2020-01-04', '2020-01-05'],
'value' : [34, 30, 35.6, 33.3, 38.1]}

df = pd.DataFrame(dataset).set_index('date')
print(df.shape)
print(df.values)
print(df)

regr6 = linear_model.BayesianRidge()
obj_MTS = ns.MTS(regr6, lags = 1, n_hidden_features=5)
obj_MTS.fit(df)
print(obj_MTS.predict())
print("\n")
print(obj_MTS.predict(return_std=True))


obj_MTS = ns.MTS(RandomForestRegressor(), lags = 1, n_hidden_features=5)
obj_MTS.fit(df)
print(obj_MTS.predict())
print("\n")


print("\n")
print("example 3 with dataframes ----- \n")

dataset = {
'date' : ['2001-01-01', '2002-01-01', '2003-01-01', '2004-01-01', '2005-01-01'],
'series1' : [34, 30, 35.6, 33.3, 38.1],    
'series2' : [4, 5.5, 5.6, 6.3, 5.1],
'series3' : [100, 100.5, 100.6, 100.2, 100.1]}
df = pd.DataFrame(dataset).set_index('date')
print(df)
print(df.columns)

# Adjust Bayesian Ridge
regr5 = linear_model.BayesianRidge()
obj_MTS = ns.MTS(regr5, lags = 2, n_hidden_features=5)
obj_MTS.fit(df)
print(obj_MTS.predict()) 
# with credible intervals
print(obj_MTS.predict(return_std=True, level=80))
print(obj_MTS.predict(return_std=True, level=95))
print(obj_MTS.predict())
