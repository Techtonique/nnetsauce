import numpy as np
from sklearn import datasets, linear_model, gaussian_process
import unittest as ut
import nnetsauce as ns


# Basic tests


class TestCustom(ut.TestCase):
    def test_custom(self):

        np.random.seed(123)
        X, y = datasets.make_regression(
            n_samples=15, n_features=3
        )
        
        breast_cancer = datasets.load_breast_cancer()
        Z = breast_cancer.data
        t = breast_cancer.target

        regr = linear_model.BayesianRidge()
        regr2 = gaussian_process.GaussianProcessClassifier()

        fit_obj = ns.Custom(
            obj=regr,
            n_hidden_features=10,
            direct_link=False,
            bias=False,
            nodes_sim="sobol",
            activation_name="relu",
            n_clusters=0,
        )

        fit_obj2 = ns.Custom(
            obj=regr,
            n_hidden_features=9,
            direct_link=False,
            bias=True,
            nodes_sim="halton",
            activation_name="sigmoid",
            n_clusters=2,
        )

        fit_obj3 = ns.Custom(
            obj=regr,
            n_hidden_features=8,
            direct_link=True,
            bias=False,
            nodes_sim="uniform",
            activation_name="tanh",
            n_clusters=3,
        )

        fit_obj4 = ns.Custom(
            obj=regr,
            n_hidden_features=7,
            direct_link=True,
            bias=True,
            nodes_sim="hammersley",
            activation_name="elu",
            n_clusters=4,
        )
        
        fit_obj5 = ns.Custom(
            obj=regr2,
            n_hidden_features=3,
            direct_link=True,
            bias=True,
            nodes_sim="hammersley",
            activation_name="elu",
            n_clusters=4,
        )

        index_train = range(10)
        index_test = range(10, 15)
        X_train = X[index_train, :]
        y_train = y[index_train]
        X_test = X[index_test, :]
        y_test = y[index_test]

        fit_obj.fit(X_train, y_train)
        err = fit_obj.predict(X_test) - y_test
        rmse = np.sqrt(np.mean(err ** 2))

        fit_obj2.fit(X_train, y_train)
        err2 = fit_obj2.predict(X_test) - y_test
        rmse2 = np.sqrt(np.mean(err2 ** 2))

        fit_obj3.fit(X_train, y_train)
        err3 = fit_obj3.predict(X_test) - y_test
        rmse3 = np.sqrt(np.mean(err3 ** 2))

        fit_obj4.fit(X_train, y_train)
        err4 = fit_obj4.predict(X_test) - y_test
        rmse4 = np.sqrt(np.mean(err4 ** 2))
        
        fit_obj5.fit(Z[0:100,:], t[0:100])
        fit_obj5.predict(Z[105,:])
        fit_obj5.predict(Z[106,:])
        
        self.assertTrue(
            np.allclose(rmse, 64.933610490495667) \
            & np.allclose(rmse2, 12.968755131423396) \
            & np.allclose(rmse3, 26.716371782298673) \
            & np.allclose(rmse4, 33.457280982445447) \
            & np.allclose(fit_obj4.predict(X_test[0, :]), 
                          127.70497052301884) \
            & np.allclose(fit_obj5.predict(Z[105,:]), 0) \
            & np.allclose(fit_obj5.predict(Z[106,:]), 1) \
        )

    def test_score(self):

        np.random.seed(123)
        X, y = datasets.make_regression(
            n_samples=15, n_features=3
        )
        breast_cancer = datasets.load_breast_cancer()
        Z = breast_cancer.data
        t = breast_cancer.target

        regr = linear_model.BayesianRidge()
        regr2 = gaussian_process.GaussianProcessClassifier()
        regr3 = gaussian_process.GaussianProcessClassifier()

        fit_obj = ns.Custom(
            obj=regr,
            n_hidden_features=100,
            direct_link=True,
            bias=True,
            nodes_sim="sobol",
            activation_name="relu",
            n_clusters=2,
        )

        fit_obj2 = ns.Custom(
            obj=regr2,
            n_hidden_features=10,
            direct_link=True,
            dropout=0.1,
            bias=True,
            activation_name="relu",
            n_clusters=3,
        )
        
        fit_obj3 = ns.Custom(
            obj=regr,
            n_hidden_features=100,
            dropout=0.6,
            direct_link=True,
            bias=True,
            nodes_sim="sobol",
            activation_name="relu",
            n_clusters=2,
        )

        fit_obj4 = ns.Custom(
            obj=regr3,
            n_hidden_features=50,
            dropout=0.5,
            direct_link=True,
            bias=True,
            activation_name="tanh",
            n_clusters=2,
        )
        
        fit_obj.fit(X, y)
        fit_obj2.fit(Z, t)
        fit_obj3.fit(X, y)
        fit_obj4.fit(Z, t)

        self.assertTrue(np.allclose(fit_obj.score(X, y), 4846.2057110929481)             
            & np.allclose(fit_obj2.score(Z, t), 0.99648506151142358) \
            & np.allclose(fit_obj3.score(X, y), 2.1668647954067132e-11) \
            & np.allclose(fit_obj4.score(Z, t), 1.0))

    def test_crossval(self):

        breast_cancer = datasets.load_breast_cancer()
        Z = breast_cancer.data
        t = breast_cancer.target
        
        diabetes = datasets.load_diabetes()
        X = diabetes.data 
        y = diabetes.target

        regr = gaussian_process.GaussianProcessClassifier()
        regr2 = linear_model.BayesianRidge()

        # create objects Custom
        fit_obj = ns.Custom(
            obj=regr,
            n_hidden_features=100,
            direct_link=True,
            bias=True,
            activation_name="relu",
            n_clusters=0,
        )
        
        # create objects Custom
        fit_obj2 = ns.Custom(
            obj=regr2,
            n_hidden_features=10,
            direct_link=True,
            bias=True,
            activation_name="relu",
            n_clusters=0,
        )

        # 5-fold cross-validation error (classification)
        self.assertTrue(
            np.allclose(
                fit_obj.cross_val_score(Z, t, cv=5),
                np.array(
                    [
                        0.95652174,
                        0.96521739,
                        0.97345133,
                        0.94690265,
                        0.95575221,
                    ]
                ),
            ) & np.allclose(
                fit_obj2.cross_val_score(X, y, cv=5),
                np.array(
                    [
                        0.45479551,  
                        0.5410207 ,  
                        0.52001879,  
                        0.44864878,  
                        0.56238344
                    ]
                ),
            ) 
        )


if __name__ == "__main__":
    ut.main()
