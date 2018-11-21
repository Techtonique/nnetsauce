import numpy as np
import psd_check as psdx
import matrix_ops as mo
import misc as mx
from sklearn import datasets

# Basic tests

# psd_check -----
A = 2*np.eye(4)

print("psd_check ----- \n")

print("Test for isPD")
print(psdx.isPD(A))
print("\n")
print("Test for nearestPD")
print(psdx.nearestPD(A))
print("\n")


# matrix_ops -----
n_features = 4
n_samples = 10
X, y = datasets.make_regression(n_features=n_features, 
                       n_samples=n_samples, 
                       random_state=0)

print("matrix_ops ----- \n")
print(A)
print("\n")
print(mo.crossprod(A))
print(mo.tcrossprod(A))
print(mo.crossprod(A, np.eye(4)))
print(mo.tcrossprod(A, np.eye(4)))
print("\n")

# misc -----
x = {"a": 3, "b": 5}
y = {"c": 1, "d": 4}

print("misc ----- \n")
print(mx.merge_two_dicts(x, y))
print("\n")