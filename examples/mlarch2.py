"""
Example: MLARCH with two forecast horizons (short-term / long-term) and
two prediction-interval levels (80% / 95%).

Demonstrates correct usage of the fixed MLARCH class: model_mean,
model_sigma, and model_residuals are ALL ns.MTS instances (fit on their own
univariate series, forecast via predict(h=...)) -- passing a bare sklearn
regressor for any of them now raises a clear TypeError at construction time
instead of failing deep inside fit().
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeCV

import nnetsauce as ns
from nnetsauce.mts.mlarch import MLARCH  # adjust import path if needed


# --------------------------------------------------------------------------
# 1. Data: real returns via yfinance, with an offline-safe fallback so this
#    example still runs without network access.
# --------------------------------------------------------------------------
def load_returns(ticker="AAPL", period="2y", seed_fallback=0):
    try:
        import yfinance as yf
        hist = yf.Ticker(ticker).history(period=period)
        prices = hist["Close"].values
        if len(prices) < 200:
            raise ValueError(f"only {len(prices)} rows returned")
        print(f"[{ticker}] real yfinance data: {len(prices)} rows, "
              f"{hist.index[0].date()} to {hist.index[-1].date()}")
        return 100 * np.diff(np.log(prices))
    except Exception as e:
        print(f"[{ticker}] *** yfinance unavailable ({e!r}) -- "
              f"using a simulated AR-GARCH series instead ***")
        rng = np.random.default_rng(seed_fallback)
        n = 500
        eps = np.zeros(n); sigma2 = np.zeros(n); sigma2[0] = 1.0; y = np.zeros(n)
        for t in range(1, n):
            sigma2[t] = 0.03 + 0.1 * eps[t - 1] ** 2 + 0.85 * sigma2[t - 1]
            eps[t] = np.sqrt(sigma2[t]) * rng.standard_normal()
            y[t] = 0.05 * y[t - 1] + eps[t]
        return y


returns = load_returns(ticker="AAPL", period="2y")
print(f"Converted to {len(returns)} log returns")
print(f"Returns: mean={returns.mean()}, std={returns.std()}")

# --------------------------------------------------------------------------
# 2. Two horizons: forecast both from the SAME fitted model, evaluate each
#    against however much of the held-out tail it actually covers.
# --------------------------------------------------------------------------
H_SHORT = 5
H_LONG = 20
LEVELS = [80, 95]

h_max = max(H_SHORT, H_LONG)
returns_train, returns_test = returns[:-h_max], returns[-h_max:]
print(f"Train: {len(returns_train)} returns, Test: {len(returns_test)} returns")

# --------------------------------------------------------------------------
# 3. Build MLARCH. All three components are ns.MTS -- this is what the
#    __init__ guard now enforces. RidgeCV is used here deliberately: a
#    length-20 *recursive* multi-step forecast (model_sigma and
#    model_residuals both forecast forward step-by-step, feeding each
#    prediction back in as a lag) can extrapolate badly with tree-based
#    estimators (GradientBoostingRegressor, RandomForest, ...) that can't
#    extrapolate outside their training leaf ranges -- worth keeping in
#    mind if you swap the estimator here for a longer H_LONG.
# --------------------------------------------------------------------------
B = 300  # conformal replications

base_estimator = RidgeCV()
mean_model = ns.MTS(base_estimator, lags=3, show_progress=False)
model_sigma = ns.MTS(base_estimator, lags=5, type_pi="scp2-kde",
                      replications=B, show_progress=False)
model_residuals = ns.MTS(base_estimator, type_pi="scp2-kde",
                          replications=B, show_progress=False)

mlarch = MLARCH(model_mean=mean_model, model_sigma=model_sigma,
                 model_residuals=model_residuals)
mlarch.fit(returns_train)
print(f"\nFitted volatility (in-sample): "
      f"mean={mlarch.fitted_volatility_mean_:.4f}, "
      f"std={mlarch.fitted_volatility_std_:.4f}")

# --------------------------------------------------------------------------
# 4. Forecast at both horizons, both levels. predict(h=...) can be called
#    repeatedly on the same fitted model -- no need to refit per horizon.
# --------------------------------------------------------------------------
results = {}
for h in (H_SHORT, H_LONG):
    for level in LEVELS:
        preds = mlarch.predict(h=h, level=level, return_sims=True)
        y_true = returns_test[:h]
        coverage = np.mean((y_true >= preds.lower) & (y_true <= preds.upper))
        rmse = np.sqrt(np.mean((preds.mean - y_true) ** 2))
        width = np.mean(preds.upper - preds.lower)
        results[(h, level)] = preds
        print(f"h={h:>2d}  level={level:>2d}%   "
              f"coverage={coverage:.2f} (target {level/100:.2f})   "
              f"RMSE={rmse:.4f}   mean interval width={width:.4f}")

# --------------------------------------------------------------------------
# 5. Plot: one panel per horizon, both levels overlaid.
# --------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
colors = {80: "tab:orange", 95: "tab:blue"}

for ax, h in zip(axes, (H_SHORT, H_LONG)):
    y_true = returns_test[:h]
    x = np.arange(h)
    ax.plot(x, y_true, "o-", color="black", label="actual", zorder=5)

    for level in LEVELS:
        preds = results[(h, level)]
        ax.plot(x, preds.mean, "--", color=colors[level], label=f"forecast (mean)"
                 if level == LEVELS[0] else None)
        ax.fill_between(x, preds.lower, preds.upper, alpha=0.25,
                         color=colors[level], label=f"{level}% interval")

    ax.set_title(f"h = {h} ({'short-term' if h == H_SHORT else 'long-term'})")
    ax.set_xlabel("step ahead")
    ax.legend(loc="best", fontsize=8)

fig.suptitle("MLARCH forecasts: short- vs long-term horizon, 80% vs 95% intervals")
fig.tight_layout()
fig.savefig("mlarch_example.png", dpi=120)
print("\nSaved plot to mlarch_example.png")