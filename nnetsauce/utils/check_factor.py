"""Check if vector is factor."""

# Authors: T. Moudiki <thierry.moudiki@gmail.com>
#
# License: BSD 3 Clause Clear

import ctypes
import numpy as np
import os
from ctypes import c_double


dir_path = os.path.dirname(__file__)
try:
    check_factorer = ctypes.cdll.LoadLibrary(dir_path + "/check_factor.so")
except:
    check_factorer = ctypes.CDLL(dir_path + "/check_factor.so")


def is_factor2(x):

    x_ = x.tolist() if isinstance(x, np.ndarray) else x.copy()

    n = len(x_)
    x_c = (c_double * n)(*x_)  # Create ctypes pointer to underlying memory

    return check_factorer.check_factor(x_c, n)


# if __name__=="__main__":
#   x1 = [65, 66, 67, 68.5, 69]
#   x2 = [65, 66, 67, 68]
#   print(check_factor(x1))
#   print(check_factor(x2))
