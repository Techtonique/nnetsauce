import numpy as np
from numpy import linalg as la
from matrixops import crossprod, tcrossprod


# in alphabetical order
# computes beta_hat = (t(x)%*%x + lam*I)^{-1}%*%t(x)%*%y    
def beta_hat(x, y, lam = 0.1):
    # assert on dimensions
    return np.dot(inv_penalized_cov(x, lam), 
                      crossprod(x, y))


# computes (t(x)%*%x + lam*I)^{-1}
def inv_penalized_cov(x, lam = 0.1):
    # assert on dimensions
    if lam == 0:
        return la.inv(crossprod(x))
    else:
        return la.inv(crossprod(x) + lam*np.eye(x.shape[1]))

    
# beta and Sigma in Bayesian Ridge Regression 1 
# without intercept! without intercept! without intercept!        
def beta_Sigma_hat_rvfl(x, y, 
                        s=0.1, sigma=0.05, 
                        x_star = None, # check when dim = 1 # test set matrix
                        return_cov = True):
    
    s2 = s**2
    lambda_ = (sigma**2)/s2
    
    if return_cov == True:
        
        Cn = inv_penalized_cov(x, lam = lambda_)
        
        beta_hat = np.dot(Cn, crossprod(x, y))
        
        Sigma_hat = s2*(np.eye(x.shape[1]) - np.dot(Cn, crossprod(x))) 
        
        if x_star is None:
            
            return (beta_hat, 
                    Sigma_hat)

        else:
            
            return (beta_hat, 
                    Sigma_hat,
                    np.dot(x_star, beta_hat),
                    np.dot(x_star, tcrossprod(Sigma_hat, x_star)) \
                    + (sigma**2)*np.eye(x_star.shape[0]))
                    
    else:
        
        if x_star is None:
            
            return beta_hat(x, y, 
                            lam = lambda_)
            
        else: 
            
            beta_hat = beta_hat(x, y, 
                             lam = lambda_)
            
            return (beta_hat, 
                    np.dot(x_star, beta_hat))
            
            
# beta and Sigma in Bayesian Ridge Regression 2
# without intercept! without intercept! without intercept!
def beta_Sigma_hat_rvfl2(x, y, 
                         Sigma, sigma=0.05,
                         x_star = None, # check when dim = 1
                         return_cov = True):
    
    Cn = la.inv(np.dot(Sigma, crossprod(x)) + 
                (sigma**2)*np.eye(x.shape[1]))
    
    temp = np.dot(Cn, tcrossprod(Sigma, x))
    
    if return_cov == True:
        
        return (np.dot(temp, y), 
                Sigma - np.dot(temp, np.dot(x, Sigma)))
        
    else:
        
        return np.dot(temp, y)
        