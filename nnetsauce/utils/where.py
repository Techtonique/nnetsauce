"""Find index of element in vector."""

# Authors: T. Moudiki <thierry.moudiki@gmail.com>
#
# License: BSD 3 Clause Clear

import ctypes
import numpy as np
import os
from ctypes import c_double, c_long


dir_path = os.path.dirname(__file__)
try:
    wherer = ctypes.cdll.LoadLibrary(dir_path + "/wherer.so")
except:
    wherer = ctypes.CDLL(dir_path + "/wherer.so")


def index_where(x, elt):
    x_ = x.tolist() if isinstance(x, np.ndarray) else x.copy()

    n = len(x_)
    x_c = (c_double * n)(*x_)  # Create ctypes pointer to underlying memory
    res = (c_long * n)(*([0] * n))

    wherer.where(x_c, c_double(elt), n, res)

    z = np.asarray(list(res))

    return z[z != -1]


# if __name__=="__main__":
#   x = [65, 68, 68, 66, 67, 68, 75, 76, 68, 68.1, 68]
#   print(index_where(x, 68))
