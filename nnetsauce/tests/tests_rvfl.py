import numpy as np
from sklearn import datasets
import unittest as ut
import nnetsauce as ns


# Basic tests

class TestRVFL(ut.TestCase):
    
    def test_rvfl(self):
        
        np.random.seed(123)
        X, y = datasets.make_regression(n_samples=25, 
                                n_features=3)
        
        fit_obj = ns.BayesianRVFL(n_hidden_features=10, 
                            direct_link=False, bias=False,
                            nodes_sim='sobol',
                            type_scaling = ('std', 'minmax', 'std'),
                            activation_name='relu', n_clusters=0)
        
        fit_obj2 = ns.BayesianRVFL2(n_hidden_features=9, 
                            direct_link=False, bias=True,
                            nodes_sim='halton',
                            type_scaling = ('std', 'minmax', 'minmax'),
                            activation_name='sigmoid', n_clusters=2)
        
        fit_obj3 = ns.BayesianRVFL(n_hidden_features=8, 
                            direct_link=True, bias=False,
                            nodes_sim='uniform',
                            type_scaling = ('minmax', 'minmax', 'std'),
                            activation_name='tanh', n_clusters=3)
        
        fit_obj4 = ns.BayesianRVFL2(n_hidden_features=7, 
                            direct_link=True, bias=True,
                            nodes_sim='hammersley',
                            type_scaling = ('minmax', 'minmax', 'minmax'),
                            activation_name='elu', n_clusters=4)
                        
        index_train = range(20)
        index_test = range(20, 25)
        X_train = X[index_train,:]
        y_train = y[index_train]
        X_test = X[index_test,:]
        y_test = y[index_test]
        
        fit_obj.fit(X_train, y_train)
        err = fit_obj.predict(X_test)[0] - y_test
        rmse = np.sqrt(np.mean(err**2))
        
        fit_obj2.fit(X_train, y_train)
        err2 = fit_obj2.predict(X_test)[0] - y_test
        rmse2 = np.sqrt(np.mean(err2**2))
        
        fit_obj3.fit(X_train, y_train)
        err3 = fit_obj3.predict(X_test)[0] - y_test
        rmse3 = np.sqrt(np.mean(err3**2))
        
        fit_obj4.fit(X_train, y_train)
        err4 = fit_obj4.predict(X_test)[0] - y_test
        rmse4 = np.sqrt(np.mean(err4**2))
        
        self.assertTrue(np.allclose(rmse, 0.81893186154747988) & \
                        np.allclose(rmse2, 17.278090150613096) & \
                        np.allclose(rmse3, 32.582378601766486) & \
                        np.allclose(rmse4, 53.052553013478331))
        
if __name__=='__main__':
    ut.main()       