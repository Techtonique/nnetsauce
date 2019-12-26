import nnetsauce as ns
import numpy as np


tscv = ns.utils.model_selection.TimeSeriesSplit()

X = np.random.rand(25, 3)
y = np.random.rand(25)


X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], 
             [9, 10], [11, 12], [13, 14]])

# Example 1 -----

for train_index, test_index in tscv.split(X, initial_window=3, 
                                          horizon=2,
                                          fixed_window=True):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    

print(tscv.get_n_splits(X, y))


# Example 2 -----

for train_index, test_index in tscv.split(X, initial_window=6,
                                          horizon=2,
                                          fixed_window=False):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    

print(tscv.get_n_splits(X, y))


# Example 3 -----

X = np.array([[1, 2], [3, 4], [1, 2], [3, 4], [1, 2], [3, 4]])
y = np.array([1, 2, 3, 4, 5, 6])

for train_index, test_index in tscv.split(X, initial_window=3, 
                                          horizon=2,
                                          fixed_window=True):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    

print(tscv.get_n_splits(X, y))


# Example 4 -----

for train_index, test_index in tscv.split(X, initial_window=2, 
                                          horizon=3,
                                          fixed_window=False):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    

print(tscv.get_n_splits(X, y))
