import numpy as np
from sklearn import datasets, linear_model, gaussian_process
import unittest as ut
import nnetsauce as ns


# Basic tests

class TestCustom(ut.TestCase):
    
    def test_custom(self):
        
        np.random.seed(123)
        X, y = datasets.make_regression(n_samples=15, 
                                n_features=3)
        
        regr = linear_model.BayesianRidge()
        
        fit_obj = ns.Custom(obj = regr, n_hidden_features=10, 
                            direct_link=False, bias=False,
                            nodes_sim='sobol',
                            activation_name='relu', n_clusters=0)
        
        fit_obj2 = ns.Custom(obj = regr, n_hidden_features=9, 
                            direct_link=False, bias=True,
                            nodes_sim='halton',
                            activation_name='sigmoid', n_clusters=2)
        
        fit_obj3 = ns.Custom(obj = regr, n_hidden_features=8, 
                            direct_link=True, bias=False,
                            nodes_sim='uniform',
                            activation_name='tanh', n_clusters=3)
        
        fit_obj4 = ns.Custom(obj = regr, n_hidden_features=7, 
                            direct_link=True, bias=True,
                            nodes_sim='hammersley',
                            activation_name='elu', n_clusters=4)
                        
        index_train = range(10)
        index_test = range(10, 15)
        X_train = X[index_train,:]
        y_train = y[index_train]
        X_test = X[index_test,:]
        y_test = y[index_test]
        
        fit_obj.fit(X_train, y_train)
        err = fit_obj.predict(X_test) - y_test
        rmse = np.sqrt(np.mean(err**2))
        
        fit_obj2.fit(X_train, y_train)
        err2 = fit_obj2.predict(X_test) - y_test
        rmse2 = np.sqrt(np.mean(err2**2))
        
        fit_obj3.fit(X_train, y_train)
        err3 = fit_obj3.predict(X_test) - y_test
        rmse3 = np.sqrt(np.mean(err3**2))
        
        fit_obj4.fit(X_train, y_train)
        err4 = fit_obj4.predict(X_test) - y_test
        rmse4 = np.sqrt(np.mean(err4**2))
        
        self.assertTrue(np.allclose(rmse, 64.933610490495667) & \
                        np.allclose(rmse2, 12.968755131423396) & \
                        np.allclose(rmse3, 26.716371782298673) & \
                        np.allclose(rmse4, 33.457280982445447))
    
    
    def test_score(self):
        
        np.random.seed(123)
        X, y = datasets.make_regression(n_samples=15, 
                                n_features=3)
        
        regr = linear_model.BayesianRidge()
        
        fit_obj = ns.Custom(obj = regr, n_hidden_features=100, 
                            direct_link=True, bias=True,
                            nodes_sim='sobol',
                            activation_name='relu', n_clusters=2)
        fit_obj.fit(X, y)
        
        self.assertTrue(np.allclose(fit_obj.score(X, y), 
                                    1.0219738235804167e-09))
    
    
    def test_crossval(self):
        
        breast_cancer = datasets.load_breast_cancer()
        Z = breast_cancer.data
        t = breast_cancer.target
        
        regr = gaussian_process.GaussianProcessClassifier()

        # create objects Custom
        fit_obj = ns.Custom(obj=regr, n_hidden_features=100, 
                             direct_link=True, bias=True,
                             activation_name='relu', n_clusters=0)
        
        # 5-fold cross-validation error (classification)
        self.assertTrue(np.allclose(fit_obj.cross_val_score(Z, t, cv = 5), 
                         np.array([0.95652174, 0.96521739, 0.97345133, 0.94690265, 0.95575221])))
        
if __name__=='__main__':
    ut.main()       