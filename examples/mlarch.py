import subprocess

try:
    subprocess.check_call(["uv", "pip", "install", "yfinance"])
except Exception as e: 
    subprocess.check_call(["pip", "install", "yfinance"])

import numpy as np
import yfinance as yf
import nnetsauce as ns 
import matplotlib.pyplot as plt 
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from statsmodels.tsa.stattools import kpss
from scipy import stats


# Define the ticker symbol
ticker_symbol = "MSFT"  # Example: Apple Inc.

# Get data for the ticker
ticker_data = yf.Ticker(ticker_symbol)

# Get the historical prices for a specific period (e.g., 1 year)
# You can adjust the start and end dates as needed
history = ticker_data.history(period="1y")

# Extract the 'Close' price series as a numpy array
# You can choose 'Open', 'High', 'Low', or 'Close' depending on your needs
stock_prices = history['Close'].values

print(f"Imported {len(stock_prices)} daily closing prices for {ticker_symbol}")
print("First 5 prices:", stock_prices[:5])
print("Last 5 prices:", stock_prices[-5:])

n_points = len(stock_prices)
h = 20
y = stock_prices[:(n_points-h)] 
y_test = stock_prices[(n_points-h):] 
n = len(y)
level=90
B=1000

plt.plot(stock_prices)

mean_model = ns.MTS(GradientBoostingRegressor(random_state=42))
model_sigma = ns.MTS(GradientBoostingRegressor(random_state=42), 
                    lags=2, type_pi="scp2-kde",
                    replications=B)
model_z = ns.MTS(GradientBoostingRegressor(random_state=42), 
                    type_pi="scp2-kde",
                    replications=B)

objMLARCH = ns.MLARCH(model_mean = mean_model,
                      model_sigma = model_sigma, 
                      model_residuals = model_z)

objMLARCH.fit(y)

preds = objMLARCH.predict(h=20, level=level)

mean_f = preds.mean
lower_bound = preds.lower
upper_bound = preds.upper

# Create a time index for the forecasts
forecast_index = np.arange(len(y), len(y) + h)
original_index = np.arange(len(y))

# Plotting
plt.figure(figsize=(12, 6))

# Plot original series
plt.plot(original_index, y, label='Original Series', color='blue')

# Plot mean forecast
plt.plot(forecast_index, mean_f, label='Mean Forecast', 
         color='red', linestyle='--')

# Plot true value
plt.plot(forecast_index, y_test, label='True test value', 
         color='green', linestyle='--')

# Plot prediction intervals
# Use the level from the results dictionary for the label
plt.fill_between(forecast_index, lower_bound, upper_bound, color='orange', 
                 alpha=0.3, label=f'{level}% Prediction Interval')

plt.title('Time Series Forecasting')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.show()