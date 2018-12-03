import numpy as np
from numpy import linalg as la
import matrixops as mo


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

    
# beta and Sigma in Bayesian Ridge Regression 1 
# without intercept! without intercept! without intercept!        
def beta_Sigma_hat_rvfl(x, y, 
                        s=0.1, sigma=0.05, 
                        fit_intercept=False,
                        x_star=None, # check when dim = 1 # check when dim = 1
                        return_cov=True):
    
    if len(x.shape) == 1:
        x = x.reshape(-1, 1)
    
    if (x_star is not None):
        if (len(x_star.shape) == 1):
            x_star = x_star.reshape(-1, 1)
    
    n, p = x.shape
    
    if fit_intercept == True:
        x = mo.cbind(np.ones(n), x)
        if x_star is not None:
            x_star = mo.cbind(np.ones(x_star.shape[0]), 
                                      x_star)
        
    s2 = s**2
    lambda_ = (sigma**2)/s2
    
    if return_cov == True:
        
        Cn = inv_penalized_cov(x, lam = lambda_)
        beta_hat_ = np.dot(Cn, mo.crossprod(x, y))
        Sigma_hat_ = s2*(np.eye(x.shape[1]) - np.dot(Cn, mo.crossprod(x))) 
        
        if x_star is None:
            
            return {'beta_hat': beta_hat_, 
                    'Sigma_hat': Sigma_hat_}

        else:
            
            return {'beta_hat': beta_hat_, 
                    'Sigma_hat': Sigma_hat_,
                    'preds': np.dot(x_star, 
                           beta_hat_),
                    'preds_std': np.sqrt(np.diag(np.dot(x_star, 
                           mo.tcrossprod(Sigma_hat_, x_star)) + \
                                    (sigma**2)*np.eye(x_star.shape[0])))
                    }
                    
    else:
        
        if x_star is None:
            
            return {'beta_hat': beta_hat(x, y, 
                            lam = lambda_)}
            
        else:
            
            beta_hat_ = beta_hat(x, y, 
                                lam = lambda_)
            return {'beta_hat': beta_hat_, 
                    'preds_std': np.dot(x_star, beta_hat_)
                    }
            
            
# beta and Sigma in Bayesian Ridge Regression 2
# without intercept! without intercept! without intercept!
def beta_Sigma_hat_rvfl2(x, y, 
                         Sigma=None, 
                         sigma=0.05,
                         fit_intercept=False,
                         x_star=None, # check when dim = 1 # check when dim = 1
                         return_cov=True):
    
    if len(x.shape) == 1:
        x = x.reshape(-1, 1)
    
    if Sigma is None:
        if fit_intercept == True:
            Sigma = np.eye(x.shape[1] + 1)
        else: 
            Sigma = np.eye(x.shape[1])
    
    if (x_star is not None):
        if (len(x_star.shape) == 1):
            x_star = x_star.reshape(-1, 1)
    
    if fit_intercept == True:
        x = mo.cbind(np.ones(x.shape[0]), x)
        if x_star is not None:
            x_star = mo.cbind(np.ones(x_star.shape[0]), 
                                      x_star)
    
    Cn = la.inv(np.dot(Sigma, mo.crossprod(x)) + \
                (sigma**2)*np.eye(x.shape[1]))
    
    temp = np.dot(Cn, mo.tcrossprod(Sigma, x))
    
    if return_cov == True:
        
        if x_star is None:       
            
            return {'beta_hat': np.dot(temp, y), 
                    'Sigma_hat': Sigma - np.dot(temp, np.dot(x, Sigma))}
            
        else:
            
            beta_hat_ = np.dot(temp, y)
            Sigma_hat_ = Sigma - np.dot(temp, np.dot(x, Sigma))
            
            return {'beta_hat': beta_hat_, 
                    'Sigma_hat': Sigma_hat_,
                    'preds': np.dot(x_star, beta_hat_), 
                    'preds_std': np.sqrt(np.diag(np.dot(x_star, 
                            mo.tcrossprod(Sigma_hat_, x_star)) + \
                            (sigma**2)*np.eye(x_star.shape[0])))
                    } 
        
    else:
        
        if x_star is None:       
            
            return {'beta_hat': np.dot(temp, y)}
        
        else:
            
            beta_hat_ = np.dot(temp, y)
            return {'beta_hat': beta_hat_,
                    'preds': np.dot(x_star, beta_hat_)
                    }