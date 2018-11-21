import numpy as np
from numpy import linalg as la
import nnetsauce.utils.matrix_ops as mo


# in alphabetical order
# computes beta_hat = (t(x)%*%x + lam*I)^{-1}%*%t(x)%*%y    
def beta_hat(x, y, lam = 0.1):
    # assert on dimensions
    return np.dot(inv_penalized_cov(x, lam), 
                      mo.crossprod(x, y))


# computes (t(x)%*%x + lam*I)^{-1}
def inv_penalized_cov(x, lam = 0.1):
    # assert on dimensions
    if lam == 0:
        return la.inv(mo.crossprod(x))
    else:
        return la.inv(mo.crossprod(x) + lam*np.eye(x.shape[1]))