import os 
import nnetsauce as ns
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn import datasets, metrics
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
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

obj1 = ns.ClassicalMTS(model="VAR")
obj1.fit(df_train)

res1 = obj1.predict(h=20) 
print(res1)
print("\n")
obj1.plot("realgovt")
obj1.plot("tbilrate")

obj1 = ns.ClassicalMTS(model="VECM")
obj1.fit(df_train)

res1 = obj1.predict(h=20) 
print(res1)
print("\n")
obj1.plot("realgovt")
obj1.plot("tbilrate")

obj1 = ns.ClassicalMTS(model="ETS")
obj1.fit(df_train['realgovt'])

res1 = obj1.predict(h=20) 
print(res1)
print("\n")
obj1.plot()

obj1 = ns.ClassicalMTS(model="ARIMA")
obj1.fit(df_train['realgovt'])

res1 = obj1.predict(h=20) 
print(res1)
print("\n")
obj1.plot()

obj1 = ns.ClassicalMTS(model="Theta")
obj1.fit(df_train['realgovt'])

res1 = obj1.predict(h=20) 
print(res1)
print("\n")
obj1.plot()
