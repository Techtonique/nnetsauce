# cython: wraparound=False
# cython: boundscheck=False
# cython: nonecheck=False

# Authors: Thierry Moudiki
#
# License: BSD 3


import numpy as np
cimport numpy as np
cimport cython
from collections import Counter


# 0 - utils -----

# 0 - 0 data structures & funcs -----

ctypedef fused nparray_double:    
    double

cdef list flatten(x): 
            
    cdef list res = []    
    
    for elt1 in x:
                    
        try:
            
            for elt2 in elt1:
                
                res.append(elt2)  
                
        except:
            
            res.append(elt1)  
                                
    return res


def is_factor(y):

    cdef long int n = len(y)
    cdef long int idx = 0
    ans = True    
    
    # check if x is int
    def is_int(x):
        try:
            return int(x) == x
        except:
            return False

    # check if x is float
    def is_float(x):
        return isinstance(x, float)
    
    for idx in range(n):
        
        if is_int(y[idx]) & (is_float(y[idx]) == False):
            
            idx += 1
            
        else:
            
            ans = False
            
            break

    return ans


# 1 - stratified subsampling based on the response -----

def subsamplec(y, double row_sample=0.8, int seed=123):
    
    cdef long int i, n_classes, n_obs, n_obs_out
    cdef long int[:] n_elem_classes
    cdef double[:] freqs_hist
    cdef list index = []

    assert (row_sample < 1) & (
        row_sample >= 0
    ), "'row_sample' must be < 1 and >= 0"

    n_obs = len(y)
    n_obs_out = np.ceil(n_obs * row_sample)

    # preproc -----
    if is_factor(y):

        classes, n_elem_classes = np.unique(y, return_counts=True)
        n_classes = len(classes)
        y_as_classes = y.copy()
        freqs_hist = np.zeros_like(n_elem_classes, dtype=np.float)
        
        for i in range(len(n_elem_classes)):
            freqs_hist[i] = np.float(n_elem_classes[i]) / n_obs

    else:

        h = np.histogram(y, bins="auto")
        n_elem_classes = np.asarray(h[0], dtype=np.int)
        freqs_hist = np.zeros_like(n_elem_classes, dtype=np.float)
        
        for i in range(len(n_elem_classes)):
            freqs_hist[i] = np.float(n_elem_classes[i]) / n_obs
        
        breaks = h[1]

        n_breaks_1 = len(breaks) - 1
        classes = range(n_breaks_1)
        n_classes = n_breaks_1
        y_as_classes = np.zeros_like(y, dtype=int)

        for i in classes:
            y_as_classes[(y > breaks[i]) * (y <= breaks[i + 1])] = int(i)

    # main loop ----

    np.random.seed(seed)

    for i in range(n_classes):

        bool_class_i = (y_as_classes == classes[i])               

        # index_class_i = [i for i, e in enumerate(bool_class_i) if e == True]
        index_class_i = np.asarray(np.where(bool_class_i == True)[0], dtype=np.int)

        if np.sum(bool_class_i) > 1:  # at least 2 elements in class  #i

            index.append(
                np.random.choice(
                    index_class_i,
                    size=int(n_obs_out * freqs_hist[i]),  # output size
                    replace=True,
                ).tolist()
            )

        else:  # only one element in class

            try:

                index.append(index_class_i[0])

            except:

                0
    
    return np.asarray(flatten(index))
    