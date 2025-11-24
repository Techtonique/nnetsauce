# ============================================================================
# COMPLETE EXAMPLE: Stock Returns Modeling
# ============================================================================
import numpy as np 
import pandas as pd
import nnetsauce as ns 
import numpy as np
import pandas as pd
from collections import namedtuple
from scipy.stats import norm
from sklearn.linear_model import Ridge, RidgeCV
from nnetsauce import MTS, MLARCH
import matplotlib.pyplot as plt


log_returns = pd.read_csv("https://raw.githubusercontent.com/Techtonique/datasets/refs/heads/main/time_series/multivariate/log_returns.csv")
log_returns.drop(columns=["Unnamed: 0"], inplace=True)
log_returns.index = pd.date_range(start="2024-04-24", periods=len(log_returns), freq="B")

# Convert to log returns (stationary)
print(f"Converted to {len(log_returns)} log returns")
print(f"Returns: mean={np.mean(log_returns)}, std={np.std(log_returns)}")

# Train/test split
h = 10  # Forecast horizon
returns_train = log_returns[:-h]
returns_test = log_returns[-h:]

print(f"\nTrain: {len(returns_train)} returns, Test: {len(returns_test)} returns")

# Create models with better regularization
mean_model = MTS(obj=Ridge(alpha=0.1), lags=5)
sigma_model = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0])  # Cross-validate
resid_model = MTS(obj=Ridge(alpha=0.1), lags=3)

# Fit MLARCH on returns
mlarch = MLARCH(mean_model, sigma_model, resid_model, lags_vol=10)
mlarch.fit(returns_train)

print(f"\nModel fitted:")
print(f"  Fitted volatility: {mlarch.fitted_volatility_mean_:.6f} ± {mlarch.fitted_volatility_std_:.6f}")
print(f"  Standardized residuals: mean={mlarch.z_mean_:.6f}, std={mlarch.z_std_:.6f}")

# Predict returns
forecast_returns = mlarch.predict(h=h, level=95, return_sims=False)

# Evaluate
coverage_returns = np.mean((returns_test >= forecast_returns.lower) &
                           (returns_test <= forecast_returns.upper))

mae_returns = np.mean(np.abs(returns_test - forecast_returns.mean))

print(f"\n{'='*60}")
print(f"RESULTS")
print(f"{'='*60}")
print(f"Returns Coverage: {coverage_returns:.1%} (target: 95%)")
print(f"MAE (returns):    {mae_returns:.6f}")
print(f"\nForecast summary:")
