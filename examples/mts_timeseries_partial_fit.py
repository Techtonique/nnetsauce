import os 
import nnetsauce as ns 
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.datasets import load_diabetes
from sklearn.linear_model import ElasticNet, Ridge, LassoCV, SGDRegressor
from sklearn.model_selection import train_test_split
from statsmodels.tsa.base.datetools import dates_from_str
from nnetsauce.utils.model_selection import cross_val_score

print(f"\n ----- Running: {os.path.basename(__file__)}... ----- \n")

# some example data
mdata = sm.datasets.macrodata.load_pandas().data
# prepare the dates index
dates = mdata[['year', 'quarter']].astype(int).astype(str)
quarterly = dates["year"] + "Q" + dates["quarter"]
quarterly = dates_from_str(quarterly)
print(mdata.head())
mdata = mdata[['realgdp', 'realinv', 'cpi']]
mdata.index = pd.DatetimeIndex(quarterly)
#data = np.log(mdata).diff().dropna()
data = mdata

n = data.shape[0]
max_idx_train = np.floor(n*0.9)
training_index = np.arange(0, max_idx_train)
testing_index = np.arange(max_idx_train, n)
df_train = data.iloc[training_index,:]
df_test = data.iloc[testing_index,:]
# Adjust Bayesian Ridge
regr4 = SGDRegressor()

obj_MTS = ns.MTS(regr4, lags = 1, n_hidden_features=5, verbose = 1, replications=100)

print(df_train.tail())

obj_MTS.fit(df_train)

print(obj_MTS.predict())

obj_MTS.partial_fit(df_test.iloc[0, :])

print(obj_MTS.predict())

obj_MTS.partial_fit(df_test.iloc[1, :])

print(obj_MTS.predict())




