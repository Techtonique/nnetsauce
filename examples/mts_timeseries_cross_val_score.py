import os 
import nnetsauce as ns 
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.datasets import load_diabetes
from sklearn.linear_model import ElasticNet, Ridge, LassoCV
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
mdata = mdata[['realgovt', 'tbilrate', 'cpi']]
mdata.index = pd.DatetimeIndex(quarterly)
data = np.log(mdata).diff().dropna()

n = data.shape[0]
max_idx_train = np.floor(n*0.9)
training_index = np.arange(0, max_idx_train)
testing_index = np.arange(max_idx_train, n)
df_train = data.iloc[training_index,:]
df_test = data.iloc[testing_index,:]

print(f"df_train.shape: {df_train.shape}")
regr = Ridge()
obj_MTS = ns.MTS(regr, lags = 3, 
                 n_hidden_features=7, 
                 seed=24, verbose = 0,
                 show_progress=False)
print(obj_MTS.cross_val_score(df_train,                         
                n_jobs=None, 
                verbose = 0,
                initial_window=100,
                horizon=5,
                fixed_window=False,                         
                show_progress=True))

regr2 = ElasticNet()
obj_MTS2 = ns.MTS(regr2, lags = 3, n_hidden_features=7, 
                  replications=100, kernel='gaussian', 
                  seed=24, verbose = 0, type_pi="scp2-bootstrap", 
                  show_progress=False)
print(obj_MTS2.cross_val_score(df_train,                         
                      n_jobs=None, 
                      verbose = 0,
                      initial_window=100,
                      horizon=5,
                      fixed_window=False,                         
                      show_progress=True, 
                      scoring="coverage"))


regr3 = Ridge()
obj_MTS3 = ns.MTS(regr3, lags = 3, n_hidden_features=7, 
                  replications=100, kernel='gaussian', 
                  seed=24, verbose = 0, type_pi="scp-block-bootstrap", 
                  show_progress=False)
print(obj_MTS3.cross_val_score(df_train,                         
                      n_jobs=None, 
                      verbose = 0,
                      initial_window=100,
                      horizon=5,
                      fixed_window=False,                         
                      show_progress=True, 
                      scoring="coverage"))


