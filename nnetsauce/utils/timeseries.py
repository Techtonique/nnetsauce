"""Tools for MTS."""

# Authors: Thierry Moudiki <thierry.moudiki@gmail.com>
#
# License: BSD 3

import numpy as np

# create inputs for training from MTS (envisage other regressors in X)
# X in decreasing order (!)
# a = np.reshape(range(0, 24), (8, 3))
# create_train_inputs(a, 2)
def create_train_inputs(X, k):

    n_k = X.shape[0] - k

    z = [X[i : n_k + i, :] for i in range(1, (k + 1))]

    return (X[0:n_k, :], np.column_stack(z))


# reformat response in prediction loop
# a = np.reshape(range(0, 24), (8, 3))
# reformat_response(a, 2)
def reformat_response(X, k):

    return np.concatenate([X[i, :] for i in range(k)])


# create k lags for series x
# def create_lags(x, k):
#
#    n = len(x)
#    n_k = n - k
#    n_col = k + 1
#    res = np.zeros((n_k, n_col))
#
#    for i in range(n_k):
#        for j in range(n_col):
#            res[i, j] = x[i + j]
#
#    return res


# create inputs for training from MTS (envisage other regressors in X)
# X in decreasing order (!)
# def create_train_inputs(X, k):
#
#    n, p = X.shape
#    n_k = n - k
#
#    y = np.zeros((n_k, p))
#    xreg = np.zeros((n_k, k*p))
#
#    for j in range(p):
#
#        Y_x = create_lags(X[:,j], k)
#
#        y[:,j] = Y_x[:, 0]
#
#        for l in range(1, k):
#
#            xreg[:, j*k + (l-1)] = Y_x[:, l]
#
#        xreg[:, j*k + (k-1)] = Y_x[:, k]
#
#    return (y, xreg)


# reformat response in prediction loop
# def reformat_response(X, k):
#
#    n, p = X.shape
#    res = np.zeros((1, k*p))
#
#    for j in range(p):
#        for i in range(k):
#            res[0, j*k + i] = X[i, j]
#
#    return res
#
