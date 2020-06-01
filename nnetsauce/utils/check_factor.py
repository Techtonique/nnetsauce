"""Check if vector is factor."""

# Authors: T. Moudiki <thierry.moudiki@gmail.com>
#
# License: BSD 3 Clause Clear

import ctypes
import numpy as np
import os 
from ctypes import c_double


try:
	dir_path = os.path.dirname(__file__)
	print(f"dir_path check_factor: {dir_path}")
	check_factorer = ctypes.cdll.LoadLibrary(dir_path + "/check_factorer.so")
except:
	try:
		dir_path = ctypes.util.find_library("check_factorer")
		print(f"dir_path check_factor: {dir_path}")
		check_factorer = ctypes.cdll.LoadLibrary(dir_path + "/check_factorer.so")
	except:
		try:
			dir_path = ctypes.util.find_library("./check_factorer")
			print(f"dir_path check_factor: {dir_path}")
			check_factorer = ctypes.cdll.LoadLibrary(dir_path + "/check_factorer.so")
		except:
			check_factorer = ctypes.CDLL('./check_factorer.so')


def is_factor2(x):
  
  x_ = x.tolist() if isinstance(x, np.ndarray) else x.copy()
    
  n = len(x_)
  x_c = (c_double * n)(*x_)  #Create ctypes pointer to underlying memory

  return check_factorer.check_factor(x_c, n)


# if __name__=="__main__":
#   x1 = [65, 66, 67, 68.5, 69]     
#   x2 = [65, 66, 67, 68]     
#   print(check_factor(x1))
#   print(check_factor(x2))
