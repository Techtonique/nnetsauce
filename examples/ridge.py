import nnetsauce as ns 
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from time import time
import numpy as np

X, y = load_diabetes(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Single lambda
print("\n\n Single lambda")
model = ns.RidgeRegressor(lambda_=1.0)
start = time()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
end = time()
print(f"Time taken: {end - start} seconds")
print(f"coefs: {model.coef_}")
# Access criteria
print(f"GCV scores: {model.GCV_}")
print(f"HKB estimate: {model.HKB_}")
print(f"LW estimate: {model.LW_}")


# Multiple lambdas
print("\n\n Multiple lambdas")
model = ns.RidgeRegressor(lambda_=[0.1, 1.0, 10.0])
start = time()
model.fit(X_train, y_train)
end = time()
print(f"Time taken: {end - start} seconds")
predictions = model.predict(X_test)  # Returns predictions for each lambda
print(f"coefs: {model.coef_}")
# Access criteria
print(f"GCV scores: {model.GCV_}")
print(f"HKB estimate: {model.HKB_}")
print(f"LW estimate: {model.LW_}")

# Compare coefficients for lambda=1.0
print("\nComparing coefficients for lambda=1.0:")
print("Single lambda coefs:", model.coef_)
print("Multiple lambda coefs (middle column):", model.coef_[:, 1])
print("Max absolute difference:", np.max(np.abs(model.coef_ - model.coef_[:, 1])))

# With GPU acceleration
#model = RidgeRegressor(lambda_=1.0, backend="gpu")