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
import matplotlib.pyplot as plt 

print(f"\n ----- Running: {os.path.basename(__file__)}... ----- \n")

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
# df_train = df.iloc[0:97,]
# df_test = df.iloc[97:123,]
df_train = df.iloc[0:158,]
df_test = df.iloc[158:198,]

print(df_train.head())
print(df_train.tail())
print(df_test.head())
print(df_test.tail())

print(f"\n 3. fit ElasticNet: ------- \n")

regr3 = linear_model.ElasticNet()
obj_MTS3 = ns.MTS(regr3, lags = 3, n_hidden_features=7, 
                  replications=100, kernel='gaussian', 
                  seed=24, verbose = 1, type_pi="kde")
start = time()
obj_MTS3.fit(df_train)
print(f"Elapsed {time()-start} s")
print("\n\n")
print(f"obj_MTS3.predict(h=5): {obj_MTS3.predict(h=5, alphas=[0.5, 2.5, 5, 16.5, 25, 50])}")
print(obj_MTS3.kde_)
print(f" Predictive simulations #1 {obj_MTS3.sims_[0]}") 
print(f" Predictive simulations #2 {obj_MTS3.sims_[1]}") 
print(f" Predictive simulations #3 {obj_MTS3.sims_[2]}") 

print(f"\n 4. fit ElasticNet SCP: ------- \n")

regr3 = linear_model.ElasticNet()
obj_MTS3 = ns.MTS(regr3, lags = 3, n_hidden_features=7, 
                  replications=100, kernel='gaussian', 
                  seed=24, verbose = 1, type_pi="scp-kde")
start = time()
obj_MTS3.fit(df_train)
print(f"Elapsed {time()-start} s")
print("\n\n")
print(f"obj_MTS3.predict(h=5): {obj_MTS3.predict(h=5, alphas=[0.5, 2.5, 5, 16.5, 25, 50])}")

print(f"\n 5. ElasticNet2 SCP2: ------- \n")

regr3 = linear_model.ElasticNet()
obj_MTS3 = ns.MTS(regr3, lags = 3, n_hidden_features=7, 
                  replications=10, kernel='gaussian', 
                  seed=24, verbose = 1, type_pi="scp2-kde")
start = time()
obj_MTS3.fit(df_train.iloc[:,0])
print(f"Elapsed {time()-start} s")
print("\n\n")
print(f"obj_MTS3.predict(h=5): {obj_MTS3.predict(h=5, alphas=[0.5, 2.5, 5, 16.5, 25, 50])}")

print(f"\n 3. fit ElasticNet: ------- \n")


obj_MTS3 = ns.MTS(regr3, lags = 3, n_hidden_features=7, 
                  replications=100, kernel='gaussian', 
                  seed=24, verbose = 1, type_pi="kde")
start = time()
obj_MTS3.fit(df_train.iloc[:,0])
print(f"Elapsed {time()-start} s")
print("\n\n")
preds = obj_MTS3.predict(h=5, alphas=[0.5, 2.5, 5, 16.5, 25, 50])
print(f"obj_MTS3.predict(h=5): {preds}")
print(obj_MTS3.kde_)
print(f" Predictive simulations #1 {obj_MTS3.sims_[0]}") 
print(f" Predictive simulations #2 {obj_MTS3.sims_[1]}") 
print(f" Predictive simulations #3 {obj_MTS3.sims_[2]}") 
preds.plot()
plt.show()