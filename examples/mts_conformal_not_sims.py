import os 
import pandas as pd 
import nnetsauce as ns 
import numpy as np 
from sklearn.linear_model import Ridge
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from time import time 

print(f"\n ----- Running: {os.path.basename(__file__)}... ----- \n")

url = "https://github.com/ritvikmath/Time-Series-Analysis/raw/master/ice_cream_vs_heater.csv"

df = pd.read_csv(url)

#df.set_index('date', inplace=True)
df.set_index('Month', inplace=True)
df.index.rename('date')

print(df.shape)
print(198*0.8)
# df_train = df.iloc[0:97,]
# df_test = df.iloc[97:123,]
df_train = df.iloc[0:158,:]
df_test = df.iloc[158:198,:]

regr = ns.PredictionInterval(obj=Ridge(),
                             method="splitconformal",
                             type_split="sequential",
                             level=95,
                             seed=312)

print(df_test)
obj_MTS = ns.MTS(regr, lags = 25, n_hidden_features=10, verbose = 1)
obj_MTS.fit(df_train)
print("\n")
print(obj_MTS.fit_objs_)
print("\n")
print(obj_MTS.predict(h=10, return_pi=True))


from sklearn.base import ClassifierMixin, RegressorMixin
try: 
    from sklearn.utils import all_estimators
except ImportError:
    pass


removed_regressors = [
    "TheilSenRegressor",
    "ARDRegression",
    "CCA",
    "GaussianProcessRegressor",
    "GradientBoostingRegressor",
    "HistGradientBoostingRegressor",
    "IsotonicRegression",
    "MultiOutputRegressor",
    "MultiTaskElasticNet",
    "MultiTaskElasticNetCV",
    "MultiTaskLasso",
    "MultiTaskLassoCV",
    "OrthogonalMatchingPursuit",
    "OrthogonalMatchingPursuitCV",
    "PLSCanonical",
    "PLSRegression",
    "RadiusNeighborsRegressor",
    "RegressorChain",
    "StackingRegressor",
    "VotingRegressor",
]

for est in all_estimators():
    if (
        issubclass(est[1], RegressorMixin)
        and (est[0] not in removed_regressors)
    ):
      try:
        print(f"Estimator: {est[0]}")        
        obj0 = ns.PredictionInterval(obj=est[1](),
                                method="splitconformal",
                                type_split="sequential",
                                level=95,
                                seed=312)
        
        regr = ns.MTS(obj=obj0,                
                lags=25)
        regr.fit(df_train)
        print(regr.predict(h=10, return_pi=True))
      except:
        pass

      try:
        print(f"Estimator: {est[0]}")        
        obj0 = ns.PredictionInterval(obj=est[1](),
                                method="localconformal",
                                type_split="sequential",
                                level=95,
                                seed=312)
        
        regr = ns.MTS(obj=obj0,                
                lags=25)
        regr.fit(df_train)
        print(regr.predict(h=10, return_pi=True))
      except Exception as e:
        pass
