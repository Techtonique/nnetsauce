import os 
import subprocess
import nnetsauce as ns
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn import datasets, metrics
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.forecasting.theta import ThetaModel
from statsmodels.tsa.base.datetools import dates_from_str

print(f"\n ----- Running: {os.path.basename(__file__)}... ----- \n")

# some example data
mdata = sm.datasets.macrodata.load_pandas().data

# prepare the dates index
dates = mdata[['year', 'quarter']].astype(int).astype(str)

quarterly = dates["year"] + "Q" + dates["quarter"]

quarterly = dates_from_str(quarterly)

mdata = mdata[['realgovt', 'tbilrate']]

mdata.index = pd.DatetimeIndex(quarterly)

data = np.log(mdata).diff().dropna()

df = data

df.index.rename('date')

idx_train = int(df.shape[0]*0.8)
idx_end = df.shape[0]
df_train = df.iloc[0:idx_train,]
df_test = df.iloc[idx_train:idx_end,]

print(df_test.head())

print("\n\n 1. VAR Model ---------- \n")
obj1 = ns.ClassicalMTS(model="VAR")
obj1.fit(df_train)

res1 = obj1.predict(h=20) 
print(res1)
print("\n")
obj1.plot("realgovt")
obj1.plot("tbilrate")

print("\n\n 2. VECM Model ---------- \n")
obj1 = ns.ClassicalMTS(model="VECM")
obj1.fit(df_train)

res1 = obj1.predict(h=20) 
print(res1)
print("\n")
obj1.plot("realgovt")
obj1.plot("tbilrate")

print("\n\n 3. ETS Model ---------- \n")
obj1 = ns.ClassicalMTS(model="ETS")
obj1.fit(df_train['realgovt'])

res1 = obj1.predict(h=20) 
print(res1)
print("\n")
obj1.plot()

print("\n\n 4. ARIMA Model ---------- \n")
obj1 = ns.ClassicalMTS(model="ARIMA")
obj1.fit(df_train['realgovt'])

res1 = obj1.predict(h=20) 
print(res1)
print("\n")
obj1.plot()

print("\n\n 5. Theta Model ---------- \n")
obj1 = ns.ClassicalMTS(model="Theta")
obj1.fit(df_train['realgovt'])

res1 = obj1.predict(h=20) 
print(res1)
print("\n")
obj1.plot()


print("\n\n 6.1 Other Model ---------- \n")
obj1 = ns.ClassicalMTS(obj=ARIMA)
obj1.fit(df_train['realgovt'], order=(0, 1, 0))
print("obj1", obj1)
res1 = obj1.predict(h=20) 
print(res1)
print("\n")
obj1.plot()

print("\n\n 6.2 Other Model2 ---------- \n")

obj1 = ns.ClassicalMTS(obj=ThetaModel)
obj1.fit(df_train['realgovt'], method="additive", period=1)
print("obj1", obj1)
res1 = obj1.predict(h=20) 
print(res1)
print("\n")
obj1.plot()

print("\n\n 6.3 Other Model3 ---------- \n")

obj1 = ns.ClassicalMTS(obj=ExponentialSmoothing)
obj1.fit(df_train['realgovt'])
print("obj1", obj1)
res1 = obj1.predict(h=20) 
print(res1)
print("\n")
obj1.plot()

