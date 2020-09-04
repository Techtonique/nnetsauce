# cython: wraparound=False
# cython: boundscheck=False
# cython: nonecheck=False

# Authors: Thierry Moudiki
#
# License: BSD 3

import functools
import numpy as np
cimport numpy as np
cimport cython
import gc
from copy import deepcopy
from tqdm import tqdm

from libc.math cimport log, exp, sqrt
from numpy.linalg import lstsq
from scipy.linalg import solve


# 0 - utils -----

# 0 - 0 data structures & funcs -----

ctypedef fused nparray_double:    
    double


cdef public double call_f(f, x):
    f(x)

    
cdef public double[:] calc_grad(f, x):
    return numerical_gradient(f, x)

    
cdef public double[:, :] calc_hessian(f, x):
    return numerical_hessian(f, x)   

DTYPE_double = np.double

DTYPE_int = np.int

ctypedef np.double_t DTYPE_double_t

def backtrack_line_search(object f, double[:] x, double[:] p_k, double[:] grad_k, 
                          double rho = 0.5, double c = 1e-4, **kwargs):
    
    cdef int i = 0
    cdef double alpha = 1.0  
    cdef int num_iters = 100
    
    f_ = lambda x: f(x, **kwargs)    
    
    for i in range(num_iters):        
        if (call_f(f_, calc_fplus(x, alpha, p_k)) <= (call_f(f_, x) + c*alpha*crossprod(grad_k, p_k))):
            break
        alpha = rho*alpha        
        
    return alpha


cdef public double calc_learning_rate(object f, double[:] x, 
                                      double[:] p_k, double[:] grad_k):
    return backtrack_line_search(f, x, p_k, grad_k)
    

def generate_index(double[:] response, double batch_prop=1.0, 
                   randomization="strat", seed=123):
    """Generation of indices for Stochastic gradient descent."""
    
    cdef long int n = len(response)
    cdef long int size_out = 0
    
    if batch_prop < 1:
    
        if randomization=="strat":        
            return subsample2(response, batch_prop, seed=seed)
        
        if randomization=="shuffle":                    
            np.random.seed(seed)        
            return np.asarray(np.floor(np.random.choice(range(n), 
                                    size=np.floor(batch_prop*n), 
                                    replace=False)), dtype=np.int)
    
    return None


cdef int check_is_min(f, double[:] x):
    # print("hessian eigen values > 0")
    # print(numerical_hessian(f, x)[1])
    return (numerical_hessian(f, x)[1])*1

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

cdef double crossprod(double[:] x, double[:] y):
    
    cdef double res 
    cdef long int i, n 
    
    n = len(y)
    res = 0 
    
    for i in range(len(x)):
        res += x[i]*y[i]
    
    return res 


cdef double[:] calc_fplus(double[:] x, double alpha, double[:] p_k):        
    
    cdef long int p = len(x)
    cdef double[:] res = np.zeros(p)     
    cdef long int j = 0
    
    for j in range(p):
        res[j] = x[j] + alpha*p_k[j]
        
    return res 

# Useful functions -----

# 1 - gradient-----
        
def numerical_gradient(object f, nparray_double[:] x, **kwargs):
    
    cdef long int p = len(x)
    cdef long int ix = 0    
    cdef double eps_factor = 6.055454452393343e-06 # machine eps ** (1/3)
    cdef double zero = 2.220446049250313e-16
    cdef double value_x, h, fx_plus, fx_minus
    cdef nparray_double[:] res = np.zeros_like(x)
    
    f_ = lambda x: f(x, **kwargs)
       
    for ix in range(p):
        
        value_x = x[ix]       
        
        h = max(eps_factor*value_x, 1e-8)
        
        x[ix] = value_x + h
        #fx_plus = f(x, **kwargs)
        fx_plus = call_f(f_, x)
        
        x[ix] = value_x - h
        #fx_minus = f(x, **kwargs)
        fx_minus = call_f(f_, x)
        
        x[ix] = value_x # restore
        
        res[ix] = (fx_plus - fx_minus)/(2*h)        
                    
    return np.asarray(res)



# 2 - hessian-----
    
def numerical_hessian(object f, nparray_double[:] x, **kwargs):
    
    cdef long int p = len(x)
    cdef long int ix, jx = 0    
    cdef double eps_factor = 0.0001220703125 # machine eps ** (1/4)
    cdef double zero = 2.220446049250313e-16
    cdef double value_x, value_y, h, k
    cdef double fx, fx_plus, fx_minus, fx_plus_minus, fx_minus_plus
    cdef double[:,::1] H = np.zeros((p, p))
    cdef double temp = 0 
    
    
    f_ = lambda x: f(x, **kwargs)
    
    fx = call_f(f_, x)
       
    for ix in range(p):
        
        for jx in range(ix, p):
            
            if (ix < jx):
                
                value_x = x[ix]            
                value_y = x[jx]
                
                h = max(eps_factor*value_x, 1e-8)
                k = max(eps_factor*value_y, 1e-8) 
                
                x[ix] = value_x + h
                x[jx] = value_y + k
                fx_plus = call_f(f_, x)
                
                x[ix] = value_x + h
                x[jx] = value_y - k
                fx_plus_minus = call_f(f_, x)
                
                x[ix] = value_x - h
                x[jx] = value_y + k
                fx_minus_plus = call_f(f_, x)
                
                x[ix] = value_x - h
                x[jx] = value_y - k
                fx_minus = call_f(f_, x)
                
                x[ix] = value_x # restore
                x[jx] = value_y # restore
                
                temp = (fx_plus - fx_plus_minus - fx_minus_plus + fx_minus)/(4*h*k)        
                                
            
            else:
                
                value_x = x[ix]                                                            
                
                h = max(eps_factor*value_x, 1e-8)
                
                x[ix] = value_x + h
                fx_plus = call_f(f_, x)
                                
                x[ix] = value_x - h
                fx_minus = call_f(f_, x)
                
                x[ix] = value_x # restore
                
                temp = (fx_plus - 2*fx + fx_minus)/(h**2)        
    
            
            H[ix, jx] = temp            
            H[jx, ix] = temp                
                
    res = np.asarray(H)
                    
    return res, all(np.linalg.eig(res)[0] > 0)


# 3 - stratified subsampling based on the response -----

def subsample2(y, double row_sample=0.8, int seed=123):
    
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


# 4 - One-hot encoder -----

def one_hot_encode(long int[:] y, 
                   int n_classes):
    
    cdef long int i 
    cdef long int n_obs = len(y)
    cdef double[:,::1] res = np.zeros((n_obs, n_classes), dtype=DTYPE_double)        

    for i in range(n_obs):
        res[i, y[i]] = 1

    return np.asarray(res)


# Coordinate descent (Stochastic) -----

# 1 - algos -----

def scd(loss_func, double[:] response, double[:] x, int num_iters=200, 
        double batch_prop=1.0, double learning_rate=0.01, double mass=0.9, 
        double decay=0.1, method="momentum", randomization="strat", 
        double tolerance=1e-3, verbose=1, 
        **kwargs):
    """Stochastic gradient descent with momentum and adaptive learning rates."""
    
    cdef int i = 0 
    cdef long int j = 0
    cdef long int n = len(response)
    cdef long int p = len(x)
    cdef double[:] velocity = np.zeros(p)
    cdef double grad_x, decay_rate, learning_rate_, h0
    cdef list losses = []

    if verbose == 1:
        iterator = tqdm(range(num_iters))          
    else:
        iterator = range(num_iters)
    
    f = lambda x: loss_func(x, **kwargs)              
       
    if method is "momentum":                

        for i in iterator: 
                   
            idx = generate_index(response=response, batch_prop=batch_prop, 
                                 randomization=randomization, 
                                 seed=i)   

            def f_j(double h, double[:] xx, long int j):        
                cdef double value_x = 0        
                cdef double res = 0 
                value_x = xx[j]            
                xx[j] = xx[j] + h
                res = loss_func(xx, row_index=idx, **kwargs)          
                xx[j] = value_x
                return res                                                                                     
            
            diff = -np.asarray(x)

            if verbose == 2:
                print(f"\n x prev: {np.asarray(x)}")
            
            for j in range(p):                          
                h0 = 6.055454452393343e-06*x[j]        
                grad_x = (f_j(h0, x, j) - f_j(-h0, x, j))/(2*h0)        
                velocity[j] = mass * velocity[j] - learning_rate * grad_x  
                x[j] = x[j] + velocity[j]        
            
            diff += np.asarray(x)  

            if verbose == 2:
                print(f"\n x new: {np.asarray(x)}")
            
            losses.append(f(x))

            if (len(losses) > 3) and (np.abs(np.diff(losses[-2:])[0]) < tolerance):
                break
            
            if verbose == 2:

                print("\n")
                print(f"iter {i+1} - decrease -----")

                try: 
                    print(np.linalg.norm(diff, 1)) 
                except:
                    pass   

                print(f"iter {i+1} - loss -----")
                print(np.flip(losses)[0])     
    
    if method in ("exp", "poly"):
        
        for i in iterator:
            
            idx = generate_index(response=response, batch_prop=batch_prop, 
                                 randomization=randomization, 
                                 seed=i)
            
            def f_j(double h, double[:] xx, long int j):        
                cdef double value_x = 0        
                cdef double res = 0 
                value_x = xx[j]            
                xx[j] = xx[j] + h
                res = loss_func(xx, row_index=idx, **kwargs)          
                xx[j] = value_x
                return res      
                                                
            decay_rate = (1 + decay*i) if method is "poly" else exp(decay*i)
            
            diff = -np.asarray(x)

            if verbose == 2:
                print(f"\n x prev: {np.asarray(x)}")
            
            losses.append(f(x))
            
            for j in range(p):  
                h0 = 6.055454452393343e-06*x[j]        
                grad_x = (f_j(h0, x, j) - f_j(-h0, x, j))/(2*h0)        
                x[j] = x[j] - grad_x*learning_rate/decay_rate
                
            diff += np.asarray(x) 

            if verbose == 2:
                print(f"\n x new: {np.asarray(x)}") 
            
            if (len(losses) > 3) and (np.abs(np.diff(losses[-2:])[0]) < tolerance):
                break

            if verbose == 2:

                print("\n")
                print(f"iter {i+1} - decrease -----")

                try:
                    print(np.linalg.norm(diff, 1))   
                except:
                    pass 

                print(f"iter {i+1} - loss -----")
                print(np.flip(losses)[0])

                    
    return np.asarray(x), num_iters, losses


# Gradient descent (Stochastic) -----  

def sgd(loss_func, double[:] response, double[:] x, int num_iters=200, 
        double batch_prop=1.0, double learning_rate=0.01, double mass=0.9, 
        double decay=0.1, method="momentum", randomization="strat", 
        double tolerance=1e-3, verbose=1, 
        **kwargs):
    """Stochastic gradient descent with momentum and adaptive learning rates."""
    
    cdef int i = 0 
    cdef long int j = 0
    cdef long int n = len(response)
    cdef long int p = len(x)
    cdef double[:] velocity = np.zeros(p)
    cdef double[:] grad_i = np.zeros(p)
    cdef double decay_rate, learning_rate_
    cdef list losses = []
    
    if verbose == 1:
        iterator = tqdm(range(num_iters))          
    else:
        iterator = range(num_iters)
            
    f = lambda x: loss_func(x, **kwargs)                            
    
    if method is "momentum":                
    
        for i in iterator:
            
            idx = generate_index(response=response, batch_prop=batch_prop, 
                                 randomization=randomization, 
                                 seed=i)
            
            def objective(double[:] x):
                return loss_func(x, row_index=idx, **kwargs)
            
            #grad_i = numerical_gradient(objective, x)                        
            grad_i = calc_grad(objective, x)    
            
            diff = -np.asarray(x)

            if verbose == 2:
                print(f"\n x prev: {np.asarray(x)}")

            for j in range(p):                
                velocity[j] = mass * velocity[j] - learning_rate * grad_i[j]                        
                x[j] = x[j] + velocity[j]
            
            diff += np.asarray(x)
            
            if verbose == 2:
                print(f"\n x new: {np.asarray(x)}")

            losses.append(f(x))

            if (len(losses) > 3) and (np.abs(np.diff(losses[-2:])[0]) < tolerance):
                break
            
            if verbose == 2:       

                print("\n")
                print(f"iter {i+1} - decrease -----")

                try:
                    print(np.linalg.norm(diff, 1))                            
                except:
                    pass  

                print(f"iter {i+1} - loss -----")
                print(np.flip(losses)[0])

    
    if method in ("exp", "poly"):
        
        for i in iterator:
            
            idx = generate_index(response=response, batch_prop=batch_prop, 
                                 randomization=randomization, 
                                 seed=i)
            
            def objective(double[:] x):
                return loss_func(x, row_index=idx, **kwargs)
            
            #grad_i = numerical_gradient(objective, x)
            grad_i = calc_grad(objective, x)            
                        
            decay_rate = (1 + decay*i) if method is "poly" else exp(decay*i)
            
            diff = -np.asarray(x)

            if verbose == 2:
                print(f"\n x prev: {np.asarray(x)}")
            
            for j in range(p):   

                x[j] = x[j] - grad_i[j]*learning_rate/decay_rate

            if verbose == 2:
                print(f"\n x new: {np.asarray(x)}")

            diff += np.asarray(x)  
            
            losses.append(f(x))

            if (len(losses) > 3) and (np.abs(np.diff(losses[-2:])[0]) < tolerance):
                break
        
            if verbose == 2:

                print("\n")
                print(f"iter {i+1} - decrease -----")
                
                try:
                    print(np.linalg.norm(diff, 1))   
                except:
                    pass
                
                print(f"iter {i+1} - loss -----")
                print(np.flip(losses)[0])
    
    return np.asarray(x), num_iters, losses
