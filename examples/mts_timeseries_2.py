import nnetsauce as ns
import numpy as np
import pandas as pd
from scipy import stats
from sklearn import datasets, metrics
from sklearn import linear_model
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.ensemble import RandomForestRegressor
from statsmodels.stats.diagnostic import acorr_ljungbox
from time import time

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

regr = linear_model.BayesianRidge()
obj_MTS = ns.MTS(regr, lags = 1, n_hidden_features=5, verbose = 1)
obj_MTS.fit(df_train.values)
print("\n")
print(obj_MTS.predict(h=5, return_std=True))
# print(f" stats.describe(obj_MTS.residuals_, axis=0, bias=False) \n {stats.describe(obj_MTS.residuals_, axis=0, bias=False)} ")
# print([acorr_ljungbox(obj_MTS.residuals_[:,i], boxpierce=True, auto_lag=True, return_df=True) for i in range(obj_MTS.residuals_.shape[1])])

regr2 = linear_model.ARDRegression()
obj_MTS2 = ns.MTS(regr2, lags = 1, n_hidden_features=5, 
                  replications=10, kernel='gaussian', 
                  seed=2324, verbose = 1)
start = time()
obj_MTS2.fit(df_train.values)
print(f"Elapsed {time()-start} s")
print("\n\n")
print(obj_MTS2.get_params())
print("\n\n")
print(obj_MTS2.kde_)
print("\n\n")
print(obj_MTS2.predict(h=5, return_std=True))
print("\n\n")
print(f"------- obj_MTS2.residuals_: {obj_MTS2.residuals_}")
print("\n\n")
print(f"------- obj_MTS2.residuals_sims_[0].shape: {obj_MTS2.residuals_sims_[0].shape}")
print("\n\n")
print(f"------- obj_MTS2.residuals_sims_[0]: {obj_MTS2.residuals_sims_[0]}")
print("\n\n")
print(f"------- obj_MTS2.residuals_sims_[0]: {obj_MTS2.residuals_sims_[0].shape}")
print("\n\n")
print(f" stats.describe(obj_MTS2.residuals_, axis=0, bias=False) \n {stats.describe(obj_MTS.residuals_, axis=0, bias=False)} ")
print("\n\n")



regr3 = linear_model.ElasticNet()
# regr3 = linear_model.Ridge()
# regr3 = RandomForestRegressor()
obj_MTS3 = ns.MTS(regr3, lags = 3, n_hidden_features=7, 
                  replications=10, kernel='gaussian', 
                  seed=24, verbose = 1)
start = time()
obj_MTS3.fit(df_train.values)
print(f"Elapsed {time()-start} s")
print("\n\n")
print(obj_MTS3.predict(h=5))
print(f" Predictive simulations #1 {obj_MTS3.sims_[0]}") 
print(f" Predictive simulations #2 {obj_MTS3.sims_[1]}") 
print(f" Predictive simulations #3 {obj_MTS3.sims_[2]}") 
print("\n\n")

# print(obj_MTS2.residuals_sims_.shape)
# print("\n\n")
# print(stats.describe(obj_MTS2.residuals_[:,0], bias=False))
# print("\n\n")
# print(stats.describe(obj_MTS2.residuals_sims_[:,0], bias=False))
# print("\n\n")
# histogram_residuals = np.histogram(obj_MTS2.residuals_[:,0])
# print(histogram_residuals)
# print("\n\n")
# print(np.histogram(obj_MTS2.residuals_sims_[:,0], bins=histogram_residuals[1]))
# print(f" stats.describe(obj_MTS2.residuals_, axis=0, bias=False) \n {stats.describe(obj_MTS2.residuals_, axis=0, bias=False)} ")
# print([acorr_ljungbox(obj_MTS2.residuals_[:,i], boxpierce=True, auto_lag=True, return_df=True) for i in range(obj_MTS2.residuals_.shape[1])])

# regr3 = GaussianProcessRegressor()
# obj_MTS3 = ns.MTS(regr3, lags = 1, n_hidden_features=5)
# obj_MTS3.fit(df_train.values)
# print(obj_MTS3.get_params())
# print("\n")
# print(obj_MTS3.predict(h=5, return_std=True))
# print(f" stats.describe(obj_MTS3.residuals_, axis=0, bias=False) \n {stats.describe(obj_MTS3.residuals_, axis=0, bias=False)} ")

