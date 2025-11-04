import os 
import nnetsauce as ns
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn import datasets, metrics
from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.base.datetools import dates_from_str
from matplotlib import pyplot as plt

def extract_month_year(df, date_column='date'):
    """
    Extracts month and year from a date column in a DataFrame.

    Parameters:
    df (pd.DataFrame): The input DataFrame containing a date column.
    date_column (str): The name of the date column to extract month and year from.

    Returns:
    pd.DataFrame: A DataFrame with the original data and new 'month' and 'year' columns.
    """
    df_feat = df.copy()
    # Ensure the date column is in datetime format
    df_feat[date_column] = pd.to_datetime(df_feat[date_column])
    
    # Extract month and year
    df_feat['month'] = df_feat[date_column].dt.month
    df_feat['year'] = df_feat[date_column].dt.year
    
    return df_feat[['month', 'year']]

print(f"\n ----- Running: {os.path.basename(__file__)}... ----- \n")

# Example 

print("--------------------------------", "realgovt")

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
h = 30

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
               replications=5,
               lags=25,
               type_pi="scp2-kde",
               kernel='gaussian',
               verbose=1)
model.fit(df_train, xreg=xreg_train)

# Predict
predictions = model.predict(h=h)
print(predictions)

model.plot("realgovt", type_plot="pi")

print("--------------------------------", "ice_cream_vs_heater")

url = "https://raw.githubusercontent.com/Techtonique/datasets/main/time_series/multivariate/ice_cream_vs_heater.csv"
df_temp = pd.read_csv(url)
df_feat = extract_month_year(df_temp)
df_feat.index = pd.DatetimeIndex(df_temp.date)
print(df_feat.head())
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
model = ns.MTS(RidgeCV(alphas=10**np.linspace(-10, 10, 100)), 
               replications=10,
               lags=25,
               type_pi="scp2-kde",
               kernel='gaussian',
               verbose=1)
model.fit(df_train, xreg=xreg_train)

predictions = model.predict(h=h)
print(predictions)

model.plot("heater", type_plot="pi")