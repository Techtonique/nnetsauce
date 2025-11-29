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
forecast_mv = stacker_mv.predict(h=5, level=90)

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