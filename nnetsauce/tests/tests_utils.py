import numpy as np
from sklearn import datasets
import nnetsauce.utils.psdcheck as psdx
import nnetsauce.utils.matrixops as mo
import nnetsauce.utils.misc as mx
import nnetsauce.utils.lmfuncs as lmf
import nnetsauce.utils.timeseries as ts 
import unittest as ut


# Basic tests

class TestUtils(ut.TestCase):

    
    # 1 - tests matrixops
    
    def test_crossprod(self):
        A = 2*np.eye(4)
        mo.crossprod(A)
        self.assertTrue(np.allclose(mo.crossprod(A), 
                                4*np.eye(4)))

    
    def test_tcrossprod(self):
        A = np.array([[ 1,  2],
                      [ 3,  4]])
        B = np.array([[ 4,  3],
                      [ 2,  1]])         
        self.assertTrue(np.allclose(mo.tcrossprod(A, B), 
                         np.array([[10,  4],
                                   [24, 10]]))) 

    
    def test_cbind(self): 
        A = np.array([[ 1,  2],
                      [ 3,  4]])
        B = np.array([[ 4,  3],
                      [ 2,  1]])         
        self.assertTrue(np.allclose(mo.cbind(A, B), 
                                np.array([[1, 2, 4, 3],
                                          [3, 4, 2, 1]]))) 
    
    
    def test_rbind(self): 
        A = np.array([[ 1,  2],
                      [ 3,  4]])
        B = np.array([[ 4,  3],
                      [ 2,  1]])         
        self.assertTrue(np.allclose(mo.rbind(A, B), 
                                np.array([[1, 2], 
                                          [3, 4],
                                          [4, 3], 
                                          [2, 1]])))
        
        
    # 2 - tests misc
    
    def test_merge_two_dicts(self):
        x = {"a": 3, "b": 5}
        y = {"c": 1, "d": 4}
        self.assertEqual(mx.merge_two_dicts(x, y),
                                {'a': 3, 'b': 5, 'c': 1, 'd': 4})


    # 3 - psd_check -----
    
    def test_psd_check(self):
        A = 2*np.eye(4)
        self.assertTrue(psdx.isPD(A))
        
        
    def test_nearestPD(self):
        A = np.array([[ 1,  2],
                      [ 3,  4]])
        self.assertTrue(np.allclose(psdx.nearestPD(A),
                        np.array([[ 1.31461828,  2.32186616],
                                  [ 2.32186616,  4.10085767]])))  

    
    # 4 - lm
    
    def test_inv_penalized_cov(self):
        X, y = datasets.make_regression(n_samples=5, 
                                n_features=2,
                                random_state=123)
        self.assertTrue(np.allclose(lmf.inv_penalized_cov(X, lam = 0.1), 
                         np.array([[0.1207683,  0.04333116],
                                   [0.04333116, 0.15787413]])))
    
    
    def test_lmf_beta_hat(self):
        X, y = datasets.make_regression(n_samples=5, 
                                n_features=2,
                                random_state=123)
        self.assertTrue(np.allclose(lmf.beta_hat(X, y, lam = 0.1), 
                         np.array([ 43.30170911,   5.68353528])))
    
    
    # 5 - MTS  
    
    def test_MTS_train_inputs(self):        
        np.random.seed(123)
        X = np.random.rand(5, 2)
        self.assertEqual(ts.create_train_inputs(X, 2)[1][1, 1],
                         0.42310646012446096)

    
    def test_MTS_reformat_response(self):
        np.random.seed(123)
        X = np.random.rand(5, 2)
        self.assertAlmostEqual(ts.reformat_response(X, 2)[1], 
                         0.28613933495)


    # 6 - lm_funcs
    
    def test_beta_Sigma_hat_rvfl(self):
        
        sigma = 0.3 
        s = 4
        np.random.seed(123)
        X, y = datasets.make_regression(n_samples=10, 
                                n_features=3)
        
        fit1 = lmf.beta_Sigma_hat_rvfl(X = X, y = y,
                              s = s, sigma = sigma,
                              fit_intercept = True,
                              return_cov = True)
        
        fit2 = lmf.beta_Sigma_hat_rvfl(X = X, y = y,
                              s = s, sigma = sigma,
                              fit_intercept = True,
                              return_cov = False)
        
        fit3 = lmf.beta_Sigma_hat_rvfl(X = X, y = y,
                              s = s, sigma = sigma,
                              fit_intercept = False,
                              return_cov = True)
        
        fit4 = lmf.beta_Sigma_hat_rvfl(X = X, y = y,
                              s = s, sigma = sigma,
                              fit_intercept = False,
                              return_cov = False)
        
        fit5 = lmf.beta_Sigma_hat_rvfl(X = X, y = y,
                                       X_star = X,
                              s = s, sigma = sigma,
                              fit_intercept = True,
                              return_cov = True)
        
        fit6 = lmf.beta_Sigma_hat_rvfl(X = X, y = y,
                                       X_star = X,
                              s = s, sigma = sigma,
                              fit_intercept = True,
                              return_cov = False)
        
        fit7 = lmf.beta_Sigma_hat_rvfl(X = X, y = y,
                                       X_star = X,
                              s = s, sigma = sigma,
                              fit_intercept = False,
                              return_cov = True)
        
        fit8 = lmf.beta_Sigma_hat_rvfl(X = X, y = y,
                                       X_star = X,
                              s = s, sigma = sigma,
                              fit_intercept = False,
                              return_cov = False)
        
        self.assertTrue(np.allclose(fit1['GCV'], fit2['GCV']) & \
                        np.allclose(fit3['GCV'], fit4['GCV']) & \
                        np.allclose(fit1['Sigma_hat'][2, 2], 0.011838759073782512) & \
                        np.allclose(fit3['Sigma_hat'][2, 2], 0.0051354540166315132) & \
                        np.allclose(fit5['beta_hat'], fit6['beta_hat']) & \
                        np.allclose(fit7['beta_hat'], fit8['beta_hat']) & \
                        np.allclose(fit5['preds_std'][0], 0.35014483) & \
                        np.allclose(fit7['preds_std'][0], 0.34927927))
        
        
    def test_beta_Sigma_hat_rvfl2(self):
        
        sigma = 0.3 
        np.random.seed(123)
        X, y = datasets.make_regression(n_samples=10, 
                                n_features=3)
        
        fit1 = lmf.beta_Sigma_hat_rvfl2(X = X, y = y,
                               sigma = sigma,
                              fit_intercept = True,
                              return_cov = True)
        
        fit2 = lmf.beta_Sigma_hat_rvfl2(X = X, y = y,
                               sigma = sigma,
                              fit_intercept = True,
                              return_cov = False)
        
        fit3 = lmf.beta_Sigma_hat_rvfl2(X = X, y = y,
                               sigma = sigma,
                              fit_intercept = False,
                              return_cov = True)
        
        fit4 = lmf.beta_Sigma_hat_rvfl2(X = X, y = y,
                               sigma = sigma,
                              fit_intercept = False,
                              return_cov = False)
        
        fit5 = lmf.beta_Sigma_hat_rvfl2(X = X, y = y,
                                       X_star = X,
                               sigma = sigma,
                              fit_intercept = True,
                              return_cov = True)
        
        fit6 = lmf.beta_Sigma_hat_rvfl2(X = X, y = y,
                                       X_star = X,
                               sigma = sigma,
                              fit_intercept = True,
                              return_cov = False)
        
        fit7 = lmf.beta_Sigma_hat_rvfl2(X = X, y = y,
                                       X_star = X,
                               sigma = sigma,
                              fit_intercept = False,
                              return_cov = True)
        
        fit8 = lmf.beta_Sigma_hat_rvfl2(X = X, y = y,
                                       X_star = X,
                               sigma = sigma,
                              fit_intercept = False,
                              return_cov = False)
        
        self.assertTrue(np.allclose(fit1['GCV'], fit2['GCV']) & \
                        np.allclose(fit3['GCV'], fit4['GCV']) & \
                        np.allclose(fit1['Sigma_hat'][2, 2], 0.011681492315826048) & \
                        np.allclose(fit3['Sigma_hat'][2, 2], 0.005107143375482126) & \
                        np.allclose(fit5['beta_hat'], fit6['beta_hat']) & \
                        np.allclose(fit7['beta_hat'], fit8['beta_hat']) & \
                        np.allclose(fit5['preds_std'][0], 0.34971076802716183) & \
                        np.allclose(fit7['preds_std'][0], 0.34893270739680532))

    
if __name__=='__main__':
    ut.main()        