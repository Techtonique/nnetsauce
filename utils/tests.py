import numpy as np
import psdcheck as psdx
import matrixops as mo
import misc as mx
import lmfuncs as lmf 
from sklearn import datasets

# Basic tests


# 1 - matrix_ops -----

A = 2*np.eye(4)

print("1 - matrix_ops ----- \n")
print("A = ")
print(A)
print("\n")
print("mo.crossprod(A) = ")
print(mo.crossprod(A))
print("\n")
print("mo.tcrossprod(A) = ")
print(mo.tcrossprod(A))
print("\n")
print("mo.crossprod(A, np.eye(4)) = ")
print(mo.crossprod(A, np.eye(4)))
print("\n")
print("mo.tcrossprod(A, np.eye(4)) = ")
print(mo.tcrossprod(A, np.eye(4)))
print("\n")
print("mo.cbind(A, A)")
print(mo.cbind(A, A))
print("\n")
print("mo.rbind(A, A)")
print(mo.rbind(A, A))
print("\n")

# 2 - misc -----

x = {"a": 3, "b": 5}
y = {"c": 1, "d": 4}

print("2 - misc ----- \n")
print("x")
print(x)
print("y")
print(y)
print("\n")
print(mx.merge_two_dicts(x, y))
print("\n")


# 3 - psd_check -----

print("3 - psd_check ----- \n")

print("Test for isPD")
print(psdx.isPD(A))
print("\n")
print("Test for nearestPD")
print(psdx.nearestPD(A))
print("\n")


# 4 - psd_check -----

n_features = 4
n_samples = 10
X, y = datasets.make_regression(n_features=n_features, 
                       n_samples=n_samples, 
                       random_state=0)
print("lmf.beta_hat(X, y, lam = 0.1)")
print(lmf.beta_hat(X, y, lam = 0.1))
print("\n")
print("lmf.inv_penalized_cov(X, lam = 0.1)")
print(lmf.inv_penalized_cov(X, lam = 0.1))
print("\n")