import os 
import nnetsauce as ns 
import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

print(f"\n ----- Running: {os.path.basename(__file__)}... ----- \n")

print(f"\n ----- Exemple 1 ----- \n")

url = "https://raw.githubusercontent.com/thierrymoudiki/mts-data/master/heater-ice-cream/ice_cream_vs_heater.csv"
#url = "https://raw.githubusercontent.com/selva86/datasets/master/Raotbl6.csv"

df = pd.read_csv(url)

# ice cream vs heater (I don't own the copyright)
df.set_index('Month', inplace=True) 
df.index.rename('date')
df2 = df.diff().dropna()
# gnp dataset
#df.set_index('date', inplace=True) 

#df = df.pct_change().dropna()
idx_train = int(df.shape[0]*0.8)
idx_end = df.shape[0]
df_train = df.iloc[0:idx_train,]
df_test = df.iloc[idx_train:idx_end,]

idx_train = int(df2.shape[0]*0.8)
idx_end = df2.shape[0]
df2_train = df2.iloc[0:idx_train,]
df2_test = df2.iloc[idx_train:idx_end,]

print(f"----- df_train: {df_train} -----")
print(f"----- df_train.dtypes: {df_train.dtypes} -----")

print(f"\n ----- Exemple 1 - 1 ----- \n")

regr_mts3 = ns.LazyDeepMTS(verbose=0, ignore_warnings=False, custom_metric=None,
                      lags = 4, n_hidden_features=7, n_clusters=2,
                      n_layers=3,
                      show_progress=False, preprocess=False, 
                      estimators=["ElasticNetCV", "RidgeCV"],
                      h=5, )
models, predictions = regr_mts3.fit(df_train, df_test)
model_dictionary = regr_mts3.provide_models(df_train, df_test)
print(models)
print(model_dictionary["DeepMTS(ElasticNetCV)"])

print(f"\n ----- Exemple 1 - 2 ----- \n")

regr_mts = ns.LazyDeepMTS(verbose=0, ignore_warnings=False , custom_metric=None,
                      lags = 4, n_hidden_features=7, n_clusters=2,
                      n_layers=3,
                      show_progress=False, preprocess=False,
                      h=5, )
models, predictions = regr_mts.fit(df_train, df_test)
model_dictionary = regr_mts.provide_models(df_train, df_test)
print(models)
print(model_dictionary["DeepMTS(LinearSVR)"])

print(f"\n ----- Exemple 1 - 3 ----- \n")

regr_mts3 = ns.LazyDeepMTS(verbose=0, ignore_warnings=False, custom_metric=None,
                      lags = 4, n_hidden_features=7, n_clusters=2,
                      n_layers=3,
                      replications=10, kernel="gaussian",
                      show_progress=False, preprocess=False,
                      h=5, )
models, predictions = regr_mts3.fit(df_train, df_test)
model_dictionary = regr_mts3.provide_models(df_train, df_test)
print(models)
print(models[['WINKLERSCORE', 'COVERAGE']])

print(f"\n ----- Exemple 1 - 4 ----- \n")

regr_mts3 = ns.LazyDeepMTS(verbose=0, ignore_warnings=False, custom_metric=None,
                      lags = 15, n_hidden_features=7, n_clusters=2,
                      n_layers=3,
                      replications=100, kernel="gaussian",
                      type_pi="scp2-kde",
                      show_progress=False, preprocess=False,
                      h=5, )
models, predictions = regr_mts3.fit(df2_train, df2_test)
model_dictionary = regr_mts3.provide_models(df2_train, df2_test)
print(models)
print(models[['WINKLERSCORE', 'COVERAGE']])



