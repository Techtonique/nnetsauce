import os 
import numpy as np 
import nnetsauce as ns
from sklearn.datasets import fetch_california_housing, load_diabetes
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from time import time
import matplotlib.pyplot as plt

print(f"\n ----- Running: {os.path.basename(__file__)}... ----- \n")

X, y = fetch_california_housing(return_X_y=True, as_frame=False)

# split data into training test and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.2, random_state=13)


print(f"\n Example 1 -----")

print(f"\n shapes -----")
print(f"X_train.shape: {X_train.shape}")
print(f"X_test.shape: {X_test.shape}")
print("\n")

# Requires further tuning
obj = ns.RandomFourierEstimator(
    estimator=DecisionTreeRegressor(max_depth=5),
    n_components=500,
    gamma=1.0,
    random_state=13
)

start = time()
obj.fit(X_train, y_train)
print(f"\n Elapsed: {time() - start}")

print("RMSE: ", np.sqrt(np.mean((y_test - obj.predict(X_test))**2)))

X, y = load_diabetes(return_X_y=True, as_frame=False)

# split data into training test and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.2, random_state=13)

print(f"\n Example 2 -----")    
print(f"\n shapes -----")
print(f"X_train.shape: {X_train.shape}")
print(f"X_test.shape: {X_test.shape}")
print("\n")
# Requires further tuning
obj = ns.RandomFourierEstimator(
    estimator=DecisionTreeRegressor(max_depth=5),
    n_components=500,
    gamma=1.0,
    random_state=13
)
start = time()
obj.fit(X_train, y_train)
print(f"\n Elapsed: {time() - start}")
print("RMSE: ", np.sqrt(np.mean((y_test - obj.predict(X_test))**2)))



