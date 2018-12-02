import numpy as np
import psdcheck as psdx
import matrixops as mo
import misc as mx
import lmfuncs as lmf 
from sklearn import datasets
import matplotlib.pyplot as plt  

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

diabetes = datasets.load_diabetes()

# define X and y
X = diabetes.data 
y = diabetes.target

print("lmf.beta_hat(X, y, lam = 0.1)")
print(lmf.beta_hat(X, y, lam = 0.1))
print("\n")
print("lmf.inv_penalized_cov(X, lam = 0.1)")
print(lmf.inv_penalized_cov(X, lam = 0.1))
print("\n")


# fit training set 

#n_samples = 442
#sigma = 0.1 
#s = 0.1
#
## without intercept
#
#y_mean = y.mean()
#
#beta, Sigma = lmf.beta_Sigma_hat_rvfl(X, y-y_mean, 
#                                      s = s, sigma = sigma, 
#                                      return_cov = True)
#y_hat = y.mean() + np.dot(X, beta)
#
#Sigma_hat = np.dot(X, mo.tcrossprod(Sigma, X)) + \
#(sigma**2)*np.eye(n_samples)
#
#ci_std = np.sqrt(np.diag(Sigma_hat))
#
## predict on test set 
#x = np.linspace(1, n_samples, num = n_samples)
##plt.scatter(x, logy, color='black')
#plt.fill_between(x, y_hat - 1.96*ci_std,  
#                 y_hat + 1.96*ci_std, 
#                 color = 'gray', 
#                 alpha = 0.4)
#
#plt.title('fits vs obs')
#plt.xlabel('x')
#plt.ylabel('fits')
#plt.show()