import numpy as np
import pandas as pd
from copy import deepcopy 
from nnetsauce import MTS, MTSStacker
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor

# Generate multivariate time series
np.random.seed(456)
n = 200
t = np.arange(n)
series1 = 10 + 0.3*t + 3*np.sin(2*np.pi*t/40) + np.random.normal(0, 1.5, n)
series2 = 20 - 0.2*t + 4*np.cos(2*np.pi*t/30) + np.random.normal(0, 2, n)
series3 = 15 + 0.1*t + 2*np.sin(2*np.pi*t/60) + np.random.normal(0, 1, n)

df_mv = pd.DataFrame({
    'series1': series1,
    'series2': series2,
    'series3': series3
})

# Same stacker setup
stacker_mv = MTSStacker(
    base_models=[Ridge(), Lasso(), ElasticNet()],
    meta_model=MTS(
        obj=Ridge(),
        lags=7,
        n_hidden_features=5,
        type_pi='kde',
        replications=200
    ),
    split_ratio=0.6
)

# Fit and predict
stacker_mv.fit(df_mv)
forecast_mv = stacker_mv.predict(h=5)

print("Multivariate forecast shape:", forecast_mv.mean.shape)
print("\nForecast for all series:")
print("mean:", forecast_mv.mean)
print("lower:", forecast_mv.lower)
print("upper:", forecast_mv.upper)

# ## How It Works Internally

# ### Training Phase
# ```
# Original data: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
#                 ↓ split_ratio=0.5
# Half1: [1, 2, 3, 4, 5]  →  Train Base Models
#                             ↓
# Half2: [6, 7, 8, 9, 10] ←  Base Predictions [p1, p2, p3]
#                             ↓
# Augmented: [6, 7, 8, 9, 10 | p1, p2, p3]
#                             ↓
#                    Train Meta-Model (MTS)
#                    (learns: original ← f(lags, base_preds))
# ```

# ### Prediction Phase
# ```
# Meta-Model forecasts ALL series jointly:
# - Uses lagged values of: [original, base_pred_1, base_pred_2, ...]
# - Outputs predictions for: [original, base_pred_1, base_pred_2, ...]
# - We extract: [original] only

# The magic: Meta-model learned cross-dependencies during training,
# so it implicitly uses base model knowledge without recomputing them!


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

print(f"\n ----- Running: {os.path.basename(__file__)}... ----- \n")

#subprocess.check_call([sys.executable, "-m", "pip", "install", "statsmodels"])
from statsmodels.stats.diagnostic import acorr_ljungbox

np.random.seed(1235)

# url = "https://raw.githubusercontent.com/selva86/datasets/master/Raotbl6.csv"
# url = "/Users/t/Documents/datasets/time_series/multivariate/Raotbl6.csv"
url = "https://github.com/ritvikmath/Time-Series-Analysis/raw/master/ice_cream_vs_heater.csv"

df = pd.read_csv(url)

#df.set_index('date', inplace=True)
df.set_index('Month', inplace=True)
#df = df.diff().dropna()
#df.index.rename('date')

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

print(f"\n 1. fit BayesianRidge: ------- \n")

regr = linear_model.BayesianRidge()
obj_MTS = MTSStacker(
    base_models=[Ridge(), Lasso(), ElasticNet()],
    meta_model=MTS(
        obj=Ridge(),
        type_pi='scp2-kde',
        replications=250
    ),
    split_ratio=0.6
)

obj_MTS.fit(df_train)
print("\n")
print(obj_MTS.predict(h=10))
# print(f" stats.describe(obj_MTS.residuals_, axis=0, bias=False) \n {stats.describe(obj_MTS.residuals_, axis=0, bias=False)} ")
# print([acorr_ljungbox(obj_MTS.residuals_[:,i], boxpierce=True, auto_lag=True, return_df=True) for i in range(obj_MTS.residuals_.shape[1])])
#obj_MTS.plot(series="rgnp")
obj_MTS.plot(series="ice cream")
obj_MTS.plot(series="heater")

