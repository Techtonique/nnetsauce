import numpy as np
from sklearn.linear_model import Ridge
import nnetsauce as ns

# Dummy vocab: suppose 5 "tokens", each a 3D vector (your MTS has 3 series)
vocab = np.array([
    [0.0, 0.0, 0.0],     # token 0: silence/rest
    [1.0, 0.2, 0.1],     # token 1: C4 soft
    [1.0, 0.8, 0.9],     # token 2: E4 loud
    [0.5, 0.5, 0.5],     # token 3: chord-ish
    [0.1, 0.9, 0.2]      # token 4: high note
])

# Your multivariate time series data (rows = time, cols = features/series)
X = np.random.rand(100, 3) * 2   # example data

model = ns.DiscreteTokenMTS(
    obj=Ridge(),
    vocab=vocab,
    metric='euclidean',
    return_mode='token_id',
    lags=4,
    n_hidden_features=5,
    type_pi='kde',          # still works for uncertainty
    replications=10
)

model.fit(X)

# Forecast → returns DataFrame with token_ids instead of floats
forecast_tokens = model.predict(h=12)
print(forecast_tokens)

# Or get full vectors:
model.return_mode = 'token_vector'
forecast_vectors = model.predict(h=12)
print(forecast_vectors)


# Or get full vectors:
model.return_mode = 'probs'
forecast_vectors = model.predict(h=12)
print(forecast_vectors)


