# cython: wraparound=False
# cython: boundscheck=False
# cython: nonecheck=False

# Authors: Thierry Moudiki
#
# License: BSD 3


cdef extern from "hyperbolictan.hpp":
    double tanh_impl "tanh_impl"(double) 

def py_tanh_impl(double x):
    return tanh_impl(x)