import os 
import nnetsauce as ns 
import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

print(f"\n ----- Running: {os.path.basename(__file__)}... ----- \n")

url = "https://raw.githubusercontent.com/thierrymoudiki/mts-data/master/heater-ice-cream/ice_cream_vs_heater.csv"

df = pd.read_csv(url)

# ice cream vs heater (I don't own the copyright)
df.set_index('Month', inplace=True) 
df.index.rename('date')

#df = df.pct_change().dropna()
idx_train = int(df.shape[0]*0.8)
idx_end = df.shape[0]
df_train = df.iloc[0:idx_train,]
df_test = df.iloc[idx_train:idx_end,]

print(f"----- df_train: {df_train} -----")
print(f"----- df_train.dtypes: {df_train.dtypes} -----")

regr_mts = ns.LazyMTS(verbose=0, ignore_warnings=False, custom_metric=None,
                      lags = 4, n_hidden_features=7, n_clusters=2)
models, predictions = regr_mts.fit(df_train, df_test)
model_dictionary = regr_mts.provide_models(df_train, df_test)
print(models)
print(model_dictionary["QuantileRegressor"])