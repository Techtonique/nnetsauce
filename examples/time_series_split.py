import os 
import nnetsauce as ns
import numpy as np

print(f"\n ----- Running: {os.path.basename(__file__)}... ----- \n")

tscv = ns.utils.model_selection.TimeSeriesSplit()

X = np.random.rand(25, 3)

#X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], 
#             [9, 10], [11, 12], [13, 14]])

# Example 1 -----

print(" \n Example 1 \n ")
for train_index, test_index in tscv.split(X, initial_window=3, 
                                          horizon=2,
                                          fixed_window=True):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]    

print(tscv.get_n_splits(X))


# Example 2 -----

print(" \n Example 2 \n ")
for train_index, test_index in tscv.split(X, initial_window=6,
                                          horizon=3,
                                          fixed_window=False):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]    

print(tscv.get_n_splits(X))


# Example 3 -----

print(" \n Example 3 \n ")
X = np.array([[1, 2], [3, 4], [1, 2], [3, 4], [1, 2], [3, 4]])

print(X)

for train_index, test_index in tscv.split(X, initial_window=3, 
                                          horizon=2,
                                          fixed_window=True):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]

print(tscv.get_n_splits(X))


# Example 4 -----

print(" \n Example 4 \n ")
for train_index, test_index in tscv.split(X, initial_window=2, 
                                          horizon=3,
                                          fixed_window=False):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]

print(tscv.get_n_splits(X))
