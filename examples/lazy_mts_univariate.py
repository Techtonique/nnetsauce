import os 
import nnetsauce as ns 
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.datasets import load_diabetes
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from statsmodels.tsa.base.datetools import dates_from_str

print(f"\n ----- Running: {os.path.basename(__file__)}... ----- \n")

print(f"\n ----- Example 1 ----- \n")

# some example data
mdata = sm.datasets.macrodata.load_pandas().data
# prepare the dates index
dates = mdata[['year', 'quarter']].astype(int).astype(str)
quarterly = dates["year"] + "Q" + dates["quarter"]
quarterly = dates_from_str(quarterly)
print(mdata.head())
mdata = mdata[['realgovt']]
mdata.index = pd.DatetimeIndex(quarterly)
data = np.log(mdata).diff().dropna()

n = data.shape[0]
max_idx_train = np.floor(n*0.9)
training_index = np.arange(0, max_idx_train)
testing_index = np.arange(max_idx_train, n)
df_train = data.iloc[training_index,:]
df_test = data.iloc[testing_index,:]

regr_mts4 = ns.LazyDeepMTS(verbose=1, ignore_warnings=True, custom_metric=None,
                       lags = 4, n_hidden_features=7, n_clusters=2,
                       type_pi = "kde", predictions=True,
                       replications=100, kernel="gaussian",
                       show_progress=True, preprocess=False)
models, predictions = regr_mts4.fit(df_train, df_test)
model_dictionary = regr_mts4.provide_models(df_train, df_test)
print(f"\n models: {models}")
print(f"\n predictions: {predictions}")
print(models[['WINKLERSCORE', 'COVERAGE']].head().values)
print(regr_mts4.get_best_model())

regr_mts5 = ns.LazyDeepMTS(verbose=1, ignore_warnings=True, custom_metric=None,
                       lags = 20, n_hidden_features=7, predictions=True, n_clusters=2,                       
                       show_progress=True, preprocess=False)
models, predictions = regr_mts5.fit(df_train, df_test)
model_dictionary = regr_mts5.provide_models(df_train, df_test)
print(f"\n models: {models}")
print(f"\n predictions: {predictions}")
print(models[['RMSE', 'MAE']].head().values)
print(regr_mts5.get_best_model())


mdata = sm.datasets.macrodata.load_pandas().data
# prepare the dates index
dates = mdata[['year', 'quarter']].astype(int).astype(str)
quarterly = dates["year"] + "Q" + dates["quarter"]
quarterly = dates_from_str(quarterly)
mdata = mdata[['cpi']]
mdata.index = pd.DatetimeIndex(quarterly)
data = np.log(mdata).diff().dropna()

n = data.shape[0]
max_idx_train = np.floor(n*0.9)
training_index = np.arange(0, max_idx_train)
testing_index = np.arange(max_idx_train, n)
df_train = data.iloc[training_index,:]
df_test = data.iloc[testing_index,:]

regr_mts4 = ns.LazyMTS(verbose=1, ignore_warnings=False, custom_metric=None,
                       lags = 10, n_hidden_features=7, n_clusters=2,
                       type_pi = "scp2-kde", predictions=True,
                       replications=100, kernel="gaussian",
                       show_progress=True, preprocess=False)
models, predictions = regr_mts4.fit(df_train, df_test)
model_dictionary = regr_mts4.provide_models(df_train, df_test)
print(f"\n models: {models}")
print(f"\n predictions: {predictions}")
print(models[['WINKLERSCORE', 'COVERAGE']].head().values)
print(regr_mts4.get_best_model())

regr_mts5 = ns.LazyMTS(verbose=1, ignore_warnings=False, custom_metric=None,
                       lags = 20, n_hidden_features=7, n_clusters=2,                       
                       show_progress=True, preprocess=False, predictions=True)
models, predictions = regr_mts5.fit(df_train, df_test)
model_dictionary = regr_mts5.provide_models(df_train, df_test)
print(f"\n models: {models}")
print(f"\n predictions: {predictions}")
print(regr_mts5.get_best_model())
print(models[['RMSE', 'MAE']].head().values)

