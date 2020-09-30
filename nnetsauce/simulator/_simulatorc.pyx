# cython: wraparound=False
# cython: boundscheck=False
# cython: nonecheck=False

# Authors: T. Moudiki
#
# License: BSD 3


import numpy as np
cimport numpy as np

cdef extern from "simulator.hpp":
    void i4_sobol_generate_in_cpp "i4_sobol_generate_in_cpp"(float [], int, int, int)
    void halton_sequence_in_cpp "halton_sequence_in_cpp"(double r[], int i1, int i2, int m)
    void hammersley_sequence_in_cpp "hammersley_sequence_in_cpp"(double r[], int i1, int i2, int m, int n)


def py_i4_sobol_generate(int m, int n):

    arr = np.asarray(np.repeat(0, m*n), dtype=np.float32)

    cdef float[::1] arr_memview = arr

    i4_sobol_generate_in_cpp(&arr_memview[0], m, n, 1)

    return np.transpose(np.reshape(arr, (n, m)))


def py_halton_sequence(int m, int n):  

    cdef int i1, i2

    i1 = 1
    i2 = n

    arr = np.asarray(np.repeat(0, m*(abs(i1-i2)+1)), dtype=np.double)

    cdef double[::1] arr_memview = arr

    halton_sequence_in_cpp(&arr_memview[0], i1, i2, m)

    return np.transpose(np.reshape(arr, (n, m)))


def py_hammersley_sequence(int m, int n):  

    cdef int i1, i2, base

    i1 = 1
    i2 = n
    base = 7

    arr = np.asarray(np.repeat(0, m*(abs(i1-i2)+1)), dtype=np.double)

    cdef double[::1] arr_memview = arr

    hammersley_sequence_in_cpp(&arr_memview[0], i1, i2, m, base)

    return np.transpose(np.reshape(arr, (n, m)))

