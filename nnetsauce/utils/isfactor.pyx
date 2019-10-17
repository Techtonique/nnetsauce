# Authors: Thierry Moudiki
#
# License: BSD 3

# check if x is int       
def is_int(x):    
    try:
        return int(x) == x
    except:
        return False


# check if x is float
def is_float(x):    
    return isinstance(x, float)


# check if the response contains only integers
cpdef is_factor_c(double[:] y):

    cdef long int n = len(y)
    cdef int ans = 1
    cdef long int idx = 0
    
    while (idx < n):
        if (is_int(y[idx]) & (is_float(y[idx])==False)):
            idx += 1
        else:
            ans = 0
            break            
    
    return ans