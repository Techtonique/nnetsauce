# cython: wraparound=False
# cython: boundscheck=False
# cython: nonecheck=False

# Authors: T. Moudiki
#
# License: BSD 3


import numpy as np
cimport numpy as np


cdef extern from "simulator.hpp":
    void i4_sobol_generate_in_C "i4_sobol_generate_in_C"(float [], int, int, int)


def py_i4_sobol_generate(int m, int n, int skip):

    arr = np.asarray(np.repeat(0, m*n), dtype=np.float32)

    cdef float[::1] arr_memview = arr

    i4_sobol_generate_in_C(&arr_memview[0], m, n, skip)

    return np.transpose(np.reshape(arr, (n, m)))

