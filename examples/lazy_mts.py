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
mdata = mdata[['realgovt', 'tbilrate', 'cpi']]
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
                       type_pi = "kde",
                       replications=100, kernel="gaussian",
                       show_progress=True, preprocess=False)
models, predictions = regr_mts4.fit(df_train, df_test)
model_dictionary = regr_mts4.provide_models(df_train, df_test)
print(models[['WINKLERSCORE', 'COVERAGE']])


regr_mts5 = ns.LazyDeepMTS(verbose=1, ignore_warnings=True, custom_metric=None,
                      lags = 20, n_hidden_features=7, n_clusters=2,
                      type_pi="scp2-kde", 
                      kernel="gaussian",
                      replications=100, 
                      show_progress=True, preprocess=False)
models, predictions = regr_mts5.fit(df_train, df_test)
model_dictionary = regr_mts5.provide_models(df_train, df_test)
print(models[['WINKLERSCORE', 'COVERAGE']])

regr_mts5 = ns.LazyDeepMTS(verbose=1, ignore_warnings=True, custom_metric=None,
                      lags = 20, n_hidden_features=7, n_clusters=2,
                      type_pi="scp2-block-bootstrap", 
                      kernel="tophat",
                      replications=100, 
                      show_progress=True, preprocess=False)
models, predictions = regr_mts5.fit(df_train, df_test)
model_dictionary = regr_mts5.provide_models(df_train, df_test)
print(models[['WINKLERSCORE', 'COVERAGE']])

regr_mts6 = ns.LazyDeepMTS(verbose=1, ignore_warnings=True, custom_metric=None,
                      lags = 20, n_hidden_features=7, n_clusters=2,
                      type_pi="scp2-block-bootstrap", 
                      kernel="tophat",
                      replications=100, 
                      show_progress=True, preprocess=False)
models, predictions = regr_mts6.fit(df_train, df_test)
model_dictionary = regr_mts6.provide_models(df_train, df_test)
print(models[['WINKLERSCORE', 'COVERAGE']])

print(f"\n ----- Example 2 ----- \n")

url = "https://raw.githubusercontent.com/Techtonique/datasets/main/time_series/multivariate/uschange.csv"
df = pd.read_csv(url)
df.set_index('date', inplace=True) 

n = df.shape[0]
max_idx_train = np.floor(n*0.9)
training_index = np.arange(0, max_idx_train)
testing_index = np.arange(max_idx_train, n)
df_train = df.iloc[training_index,:]
df_test = df.iloc[testing_index,:]
print(f"horizon={df_test.shape[0]}")

regr_mts4 = ns.LazyDeepMTS(verbose=1, ignore_warnings=True, custom_metric=None,
                       lags = 20, n_hidden_features=7, n_clusters=2,
                       type_pi = "scp2-block-bootstrap",
                       replications = 100, kernel="gaussian",
                       estimators = ["Ridge", "Lasso", "LarsCV", "LassoCV", "LassoLarsCV"],
                       show_progress=True, preprocess=False)
models, predictions = regr_mts4.fit(df_train, df_test)
model_dictionary = regr_mts4.provide_models(df_train, df_test)
print(models[['WINKLERSCORE', 'COVERAGE']])

regr_mts5 = ns.LazyDeepMTS(verbose=1, ignore_warnings=True, custom_metric=None,
                        lags = 20, n_hidden_features=7, n_clusters=2,
                        type_pi = "scp-bootstrap",
                        replications=100, kernel="gaussian",
                        estimators = ["Ridge", "Lasso", "LarsCV", "LassoCV", "LassoLarsCV"],
                        show_progress=False, preprocess=False)
models, predictions = regr_mts5.fit(df_train, df_test)
model_dictionary = regr_mts5.provide_models(df_train, df_test)
print(models[['WINKLERSCORE', 'COVERAGE']])

regr_mts5 = ns.LazyDeepMTS(verbose=1, ignore_warnings=True, custom_metric=None,
                    lags = 20, n_hidden_features=7, n_clusters=2,
                    type_pi = "gaussian",
                    show_progress=False, preprocess=False)
models, predictions = regr_mts5.fit(df_train, df_test)
model_dictionary = regr_mts5.provide_models(df_train, df_test)
print(models[['WINKLERSCORE', 'COVERAGE']])

