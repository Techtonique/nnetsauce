# cython: wraparound=False
# cython: boundscheck=False
# cython: nonecheck=False

# Authors: Thierry Moudiki
#
# License: BSD 3

import functools
import pickle
import numpy as np
cimport numpy as np
cimport cython
import gc

from cython.parallel cimport prange
from libc.math cimport log, exp, sqrt, fabs
from numpy.linalg import lstsq
from numpy.linalg import norm
from scipy.special import expit
from sklearn.cluster import MiniBatchKMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from multiprocessing import Pool
from tqdm import tqdm
from ..utils import Progbar
from ..utils import subsample


# 0 - utils -----


# 0 - 0 data structures & funcs -----

# a tuple of doubles
cdef struct mydoubletuple:
    double elt1
    double elt2

ctypedef fused nparray_int:
    int
    
ctypedef fused nparray_double:    
    double
    
ctypedef fused nparray_long:
    long long

cdef dict __find_kmin_x_cache = {}
    


# 0 - 1 fitting -----

# returns max(x, y)
cdef double max_c(double x, double y):
    if (x > y):
        return x 
    return y


# returns min(x, y)
cdef double min_c(double x, double y):
    if (x < y):
        return x 
    return y


# sum vector 
cdef double cython_sum(double[:] x, long int n):
    cdef double res = 0 
    cdef long int i
    for i in range(n):
        res += x[i]
    return res


# finds index of a misclassed element 
cdef long int find_misclassed_index(nparray_long[:] misclass, 
                                    long int n_obs):    
    
    cdef long int i  
    for i in range(n_obs):        
        if (misclass[i] != 1000): # a badly classified example            
            return i    
    return 100000


# calculates sumproduct or 2 vectors
cdef double sum_product(long int[:] x, double[:] y, 
                        long int n):    
    cdef double res = 0
    cdef long int i    
    for i in range(n):
        if (x[i] != 0) & (y[i] != 0):
            res += x[i]*y[i]    
    return res


# calculates tolerance for early stopping
cdef double calculate_tolerance(nparray_double[:] x):     
    
    cdef long int n    
    cdef double ans 
    cdef nparray_double[:] temp
    
    if (len(x) >= 5): 
        
        x_ = np.asarray(x)                
        temp = x_[np.where(x_ != 0.0)] 
        n = len(temp)
        
        try:
            
            ans = (temp[n-1] - temp[(n-2)])        
            if (ans > 0):
                return ans
            return -ans
        
        except:
            
            return 1e5
        
    else:
        
        return 1e5

# 1 main fitting loop -----       

def rbagloop(object base_learner, double[:,:] X, long int[:] y,
int n_estimators, int verbose, int seed):

    cdef int m 
    cdef dict voter 

    voter = dict.fromkeys(range(n_estimators))

    if verbose == 1:
        pbar = Progbar(n_estimators)

    for m in range(n_estimators):

        try:

            base_learner.fit(X, y)

            voter.update(
                {m: pickle.loads(pickle.dumps(base_learner, -1))}
            )

            base_learner.set_params(seed=seed + (m + 1) * 1000)

            if verbose == 1:
                pbar.update(m)

        except:

            if verbose == 1:
                pbar.update(m)

            continue

    if verbose == 1:
        pbar.update(n_estimators)
    
    return voter
