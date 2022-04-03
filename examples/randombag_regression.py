import numpy as np 
import nnetsauce as ns
from sklearn.datasets import fetch_california_housing
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from time import time
import matplotlib.pyplot as plt


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
obj = DecisionTreeRegressor()
obj2 = ns.RandomBagRegressor(obj = obj,
                               direct_link=True,
                               n_estimators=50,
                               n_hidden_features=5, 
                               n_clusters=3,                              
                               dropout = 0.7946672521272369,
                               col_sample = 0.7305452927102478,
                               row_sample = 0.9247504466856605,
                               type_clust = "gmm", 
                               verbose = 1)

start = time()
obj2.fit(X_train, y_train)
print(f"\n Elapsed: {time() - start}")

print("RMSE: ")
print(np.sqrt(obj2.score(X_test, y_test))) # RMSE


