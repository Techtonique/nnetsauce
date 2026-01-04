import nnetsauce as ns 
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

# Create sample multivariate time series (3 series, 100 observations)
np.random.seed(42)
n_obs = 100
data = {
    'sales': np.cumsum(np.random.randn(n_obs)) + 100,
    'revenue': np.cumsum(np.random.randn(n_obs)) + 500,
    'orders': np.cumsum(np.random.randn(n_obs)) + 50
}
df = pd.DataFrame(data)

print("Data shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

# Fit MultivariateMTS with vectorized Ridge
model = ns.MultiOutputMTS(
    obj=Ridge(alpha=1.0),
    lags=3,
    n_hidden_features=10,
    type_pi='bootstrap',
    replications=100,
    verbose=1,
    show_progress=False
)

# Fit model
model.fit(df)

# Predict 10 steps ahead with 95% prediction intervals
forecast = model.predict(h=10, level=95)
print("forecast:", forecast)
print("\n" + "="*60)
print("FORECAST RESULTS")
print("="*60)
print("\nMean predictions:")
print(forecast.mean)
print("\nLower bounds (95%):")
print(forecast.lower)
print("\nUpper bounds (95%):")
print(forecast.upper)

# Predict specific quantiles
quantile_forecast = model.predict(h=10, quantiles=[0.1, 0.5, 0.9])
print("\n" + "="*60)
print("QUANTILE PREDICTIONS")
print("="*60)
print(quantile_forecast)