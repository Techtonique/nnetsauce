import os 
import nnetsauce as ns
import numpy as np
import pandas as pd
from sklearn import datasets, metrics
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor

print(f"\n ----- Running: {os.path.basename(__file__)}... ----- \n")

url = "https://github.com/ritvikmath/Time-Series-Analysis/raw/master/ice_cream_vs_heater.csv"

df = pd.read_csv(url)

#df.set_index('date', inplace=True)
df.set_index('Month', inplace=True)
df.index.rename('date')

print(df)
print("\n")

# Adjust Ridge
regr4 = linear_model.Ridge()
obj_MTS = ns.DeepMTS(regr4, 
                     n_layers=3,
                     lags = 1, 
                     n_hidden_features=5, 
                     replications=10, 
                     kernel='gaussian', 
                     verbose = 1)
obj_MTS.fit(df)
res1 = obj_MTS.predict() 
print(res1)
print("\n")

print(f" Predictive simulations #1 {obj_MTS.sims_[0]}") 
print(f" Predictive simulations #2 {obj_MTS.sims_[1]}") 
print(f" Predictive simulations #3 {obj_MTS.sims_[2]}") 
print("\n\n")


# Adjust RandomForestRegressor
regr5 = RandomForestRegressor()
obj_MTS2 = ns.DeepMTS(regr5, 
                     n_layers=3,
                     lags = 1, 
                     n_hidden_features=5,
                     replications=10, 
                     kernel='gaussian',  
                     verbose = 1)
obj_MTS2.fit(df)

# with credible intervals
res2 = obj_MTS2.predict() 
print(res2)
print("\n")

print(f" Predictive simulations #1 {obj_MTS2.sims_[0]}") 
print(f" Predictive simulations #2 {obj_MTS2.sims_[1]}") 
print(f" Predictive simulations #3 {obj_MTS2.sims_[2]}") 
print("\n\n")

