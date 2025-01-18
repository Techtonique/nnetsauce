import os 
import nnetsauce as ns
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn import datasets, metrics
from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.base.datetools import dates_from_str
from matplotlib import pyplot as plt

print(f"\n ----- Running: {os.path.basename(__file__)}... ----- \n")

# Example 

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
h = 25

n = data.shape[0]
max_idx_train = int(np.floor(n*0.9))
training_index = np.arange(0, max_idx_train)
testing_index = np.arange(max_idx_train, n)
df_train = data.iloc[training_index,:]
print(df_train.shape)
df_test = data.iloc[testing_index,:]

# Create trend and seasonal features for the full dataset first
n_total = len(data)
trend = np.arange(n_total)
seasonal = pd.get_dummies(data.index.quarter)  # quarterly seasonality

# Split data into train/test
max_idx_train = int(np.floor(n_total*0.9))
training_index = np.arange(0, max_idx_train)
testing_index = np.arange(max_idx_train, n_total)

df_train = data.iloc[training_index]
df_test = data.iloc[testing_index]

# Split external regressors (trend and seasonality)
xreg_train = pd.DataFrame(
    np.column_stack([trend[training_index], seasonal.iloc[training_index]]),
    columns=['trend', 'Q1', 'Q2', 'Q3', 'Q4'],
    index=df_train.index  # Important: use same index as training data
)

xreg_test = pd.DataFrame(
    np.column_stack([trend[testing_index], seasonal.iloc[testing_index]]),
    columns=['trend', 'Q1', 'Q2', 'Q3', 'Q4'],
    index=df_test.index  # Important: use same index as test data
)

# Fit model
model = ns.MTS(RidgeCV(alphas=10**np.linspace(-3, 3, 100)), 
               replications=100,
               lags=25,
               type_pi="scp2-kde",
               kernel='gaussian',
               verbose=1)
model.fit(df_train, xreg=xreg_train)

# Predict
predictions = model.predict(h=h)
print(predictions)

model.plot("realgovt", type_plot="pi")


url = "https://raw.githubusercontent.com/Techtonique/datasets/main/time_series/multivariate/ice_cream_vs_heater.csv"
df_temp = pd.read_csv(url)
df_temp.index = pd.DatetimeIndex(df_temp.date)
data = df_temp.drop(columns=['date']).diff().dropna()

n = data.shape[0]
max_idx_train = int(np.floor(n*0.9))
training_index = np.arange(0, max_idx_train)
testing_index = np.arange(max_idx_train, n)
df_train = data.iloc[training_index,:]
df_test = data.iloc[testing_index,:]

# Create trend and seasonal features for the full dataset first
n_total = len(data)
trend = np.arange(n_total)
seasonal = pd.get_dummies(data.index.quarter)  # quarterly seasonality

# Split data into train/test
max_idx_train = int(np.floor(n_total*0.9))
training_index = np.arange(0, max_idx_train)
testing_index = np.arange(max_idx_train, n_total)
df_train = data.iloc[training_index]
df_test = data.iloc[testing_index]

xreg_train = pd.DataFrame(
    np.column_stack([trend[training_index], seasonal.iloc[training_index]]),
    columns=['trend', 'Q1', 'Q2', 'Q3', 'Q4'],
    index=df_train.index  # Important: use same index as training data
)

xreg_test = pd.DataFrame(
    np.column_stack([trend[testing_index], seasonal.iloc[testing_index]]),
    columns=['trend', 'Q1', 'Q2', 'Q3', 'Q4'],
    index=df_test.index  # Important: use same index as test data
)

# Fit model
model = ns.MTS(RidgeCV(alphas=10**np.linspace(-3, 3, 100)), 
               replications=100,
               lags=25,
               type_pi="scp2-kde",
               kernel='gaussian',
               verbose=1)
model.fit(df_train, xreg=xreg_train)

predictions = model.predict(h=h)
print(predictions)

model.plot("heater", type_plot="pi")


url = "https://raw.githubusercontent.com/Techtonique/datasets/main/time_series/univariate/AirPassengers.csv"
df = pd.read_csv(url)
df.index = pd.DatetimeIndex(df.date)
df.drop(columns=['date'], inplace=True)
data = df 

n = data.shape[0]
max_idx_train = int(np.floor(n*0.9))
training_index = np.arange(0, max_idx_train)
testing_index = np.arange(max_idx_train, n)
df_train = data.iloc[training_index]
df_test = data.iloc[testing_index]

trend = np.arange(n)

xreg_train = pd.DataFrame(
    trend[training_index],
    columns=['trend'],
    index=df_train.index  # Important: use same index as training data
)

xreg_test = pd.DataFrame(
    trend[testing_index],
    columns=['trend'],
    index=df_test.index  # Important: use same index as test data
)

model = ns.MTS(LassoCV(alphas=10**np.linspace(-10, 10, 100)), 
               replications=3,
               lags=15,
               type_pi="scp2-kde",
               kernel='gaussian',
               verbose=1)
model.fit(df_train, xreg=xreg_train)

predictions = model.predict(h=h)
print(predictions)

model.plot(type_plot="pi")
