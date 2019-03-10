import numpy as np
from sklearn import linear_model
import unittest as ut
import nnetsauce as ns


# Basic tests

class TestMTS(ut.TestCase):
    
    def test_MTS(self):
        
        np.random.seed(123)
        X = np.random.rand(25, 3)
        X[:,0] = 100*X[:,0]
        X[:,2] = 25*X[:,2]
        
        regr = linear_model.BayesianRidge()
        
        fit_obj = ns.MTS(regr, n_hidden_features=10, 
                            direct_link=False, bias=False,
                            nodes_sim='sobol',
                            type_scaling = ('std', 'minmax', 'std'),
                            activation_name='relu', n_clusters=0)
        
        fit_obj2 = ns.MTS(regr, n_hidden_features=9, 
                            direct_link=False, bias=True,
                            nodes_sim='halton',
                            type_scaling = ('std', 'minmax', 'minmax'),
                            activation_name='sigmoid', n_clusters=2)
        
        fit_obj3 = ns.MTS(regr, n_hidden_features=8, 
                            direct_link=True, bias=False,
                            nodes_sim='uniform',
                            type_scaling = ('minmax', 'minmax', 'std'),
                            activation_name='tanh', n_clusters=3)
        
        fit_obj4 = ns.MTS(regr, n_hidden_features=7, 
                            direct_link=True, bias=True,
                            nodes_sim='hammersley',
                            type_scaling = ('minmax', 'minmax', 'minmax'),
                            activation_name='elu', n_clusters=4)
                        
        index_train = range(20)
        index_test = range(20, 25)
        X_train = X[index_train,:]
        X_test = X[index_test,:]
        
        fit_obj.fit(X_train)
        err = fit_obj.predict() - X_test
        rmse = np.sqrt(np.mean(err**2))
        
        fit_obj2.fit(X_train)
        err2 = fit_obj2.predict() - X_test
        rmse2 = np.sqrt(np.mean(err2**2))
        
        fit_obj3.fit(X_train)
        err3 = fit_obj3.predict() - X_test
        rmse3 = np.sqrt(np.mean(err3**2))
        
        fit_obj4.fit(X_train)
        err4 = fit_obj4.predict() - X_test
        rmse4 = np.sqrt(np.mean(err4**2))
        
        self.assertTrue(np.allclose(rmse, 10.396062391967684) & \
                        np.allclose(rmse2, 10.395489235411796) & \
                        np.allclose(rmse3, 10.395986434438191) & \
                        np.allclose(rmse4, 10.677585029352571))
        
        
    def test_get_set(self):
        
        np.random.seed(123)
        X = np.random.rand(25, 3)
        X[:,0] = 100*X[:,0]
        X[:,2] = 25*X[:,2]
        
        regr = linear_model.BayesianRidge()
        
        fit_obj = ns.MTS(regr, n_hidden_features=10, 
                            direct_link=False, bias=False,
                            nodes_sim='sobol',
                            type_scaling = ('std', 'minmax', 'std'),
                            activation_name='relu', n_clusters=0)
        
        fit_obj.set_params(n_hidden_features=5, 
                   activation_name='relu', a=0.01,
                   nodes_sim='sobol', bias=True,
                   direct_link=True, n_clusters=None,
                   type_clust='kmeans', 
                   type_scaling = ('std', 'std', 'std'),
                   seed=123, 
                   lags = 1)
        
        self.assertTrue((fit_obj.get_params()['lags'] == 1) & (fit_obj.get_params()['type_scaling'] == ('std', 'std', 'std')))
        
        
    def test_score(self):
        
        np.random.seed(123)
        X = np.random.rand(25, 3)
        X[:,0] = 100*X[:,0]
        X[:,2] = 25*X[:,2]

        regr = linear_model.BayesianRidge()
        
        fit_obj = ns.MTS(regr, n_hidden_features=10, 
                            direct_link=False, bias=False,
                            nodes_sim='sobol',
                            type_scaling = ('std', 'minmax', 'std'),
                            activation_name='relu', n_clusters=0)
        
        self.assertTrue(np.allclose(fit_obj.score(X, training_index = range(20), 
                                                  testing_index = range(20, 25),
                                                  scoring='neg_mean_squared_error'), 
        (239.14320170278387, 0.080854374885662481, 85.010283695384985)))
        
if __name__=='__main__':
    ut.main()       