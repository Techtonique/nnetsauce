import numpy as np
from sklearn import linear_model, gaussian_process
import unittest as ut
import nnetsauce as ns


# Basic tests

np.random.seed(123)


class TestMTS(ut.TestCase):
    def test_MTS(self):

        np.random.seed(123)
        X = np.random.rand(25, 3)
        X[:, 0] = 100 * X[:, 0]
        X[:, 2] = 25 * X[:, 2]

        regr = linear_model.BayesianRidge()
        regr2 = gaussian_process.GaussianProcessRegressor()
        regr3 = ns.BayesianRVFLRegressor(s=0.01)
        regr4 = ns.BayesianRVFL2Regressor(s1=0.01, s2=0.01, n_hidden_features=2)

        fit_obj = ns.MTS(
            regr,
            n_hidden_features=10,
            direct_link=False,
            bias=False,
            nodes_sim="sobol",
            type_scaling=("std", "minmax", "std"),
            activation_name="relu",
            n_clusters=0,
        )

        fit_obj2 = ns.MTS(
            regr,
            n_hidden_features=9,
            direct_link=False,
            bias=True,
            nodes_sim="halton",
            type_scaling=("std", "minmax", "minmax"),
            activation_name="sigmoid",
            n_clusters=2,
        )

        fit_obj3 = ns.MTS(
            regr,
            n_hidden_features=8,
            direct_link=True,
            bias=False,
            nodes_sim="uniform",
            type_scaling=("minmax", "minmax", "std"),
            activation_name="tanh",
            n_clusters=3,
        )

        fit_obj4 = ns.MTS(
            regr,
            n_hidden_features=7,
            direct_link=True,
            bias=True,
            nodes_sim="hammersley",
            type_scaling=("minmax", "minmax", "minmax"),
            activation_name="elu",
            n_clusters=4,
        )

        fit_obj5 = ns.MTS(
            regr,
            n_hidden_features=7,
            direct_link=True,
            bias=True,
            dropout=0.5,
            nodes_sim="hammersley",
            type_scaling=("minmax", "minmax", "minmax"),
            activation_name="elu",
            n_clusters=4,
        )

        fit_obj6 = ns.MTS(
            regr2,
            n_hidden_features=2,
            direct_link=True,
            bias=True,
            dropout=0.5,
            nodes_sim="hammersley",
            type_scaling=("minmax", "minmax", "minmax"),
            activation_name="elu",
            n_clusters=4,
        )

        fit_obj7 = ns.MTS(
            regr3,
            n_hidden_features=2,
            direct_link=True,
            bias=True,
            dropout=0.5,
            nodes_sim="hammersley",
            type_scaling=("minmax", "minmax", "minmax"),
            activation_name="elu",
            n_clusters=4,
        )

        fit_obj8 = ns.MTS(
            regr4,
            n_hidden_features=0,
            direct_link=True,
            bias=True,
            dropout=0.5,
            nodes_sim="hammersley",
            type_scaling=("minmax", "minmax", "minmax"),
            activation_name="elu",
            n_clusters=4,
        )

        fit_obj9 = ns.MTS(
            regr4,
            n_hidden_features=0,
            direct_link=True,
            bias=True,
            dropout=0.5,
            nodes_sim="hammersley",
            type_scaling=("minmax", "minmax", "minmax"),
            activation_name="elu",
            n_clusters=4,
            cluster_encode=False,
        )

        fit_obj10 = ns.MTS(
            regr4,
            n_hidden_features=0,
            direct_link=True,
            bias=True,
            dropout=0.5,
            nodes_sim="hammersley",
            type_scaling=("minmax", "minmax", "minmax"),
            activation_name="elu",
            n_clusters=4,
            cluster_encode=False,
        )

        index_train = range(20)
        index_test = range(20, 25)
        X_train = X[index_train, :]
        X_test = X[index_test, :]

        Xreg_train = np.reshape(range(0, 60), (20, 3))
        Xreg_test = np.reshape(range(60, 75), (5, 3))

        fit_obj.fit(X=X_train)
        err = fit_obj.predict() - X_test
        rmse = np.sqrt(np.mean(err ** 2))

        fit_obj.fit(X_train, xreg=Xreg_train)
        err_xreg = fit_obj.predict(new_xreg=Xreg_test) - X_test
        rmse_xreg = np.sqrt(np.mean(err_xreg ** 2))

        fit_obj2.fit(X_train)
        err2 = fit_obj2.predict() - X_test
        rmse2 = np.sqrt(np.mean(err2 ** 2))

        fit_obj3.fit(X_train)
        err3 = fit_obj3.predict() - X_test
        rmse3 = np.sqrt(np.mean(err3 ** 2))

        fit_obj4.fit(X_train)
        err4 = fit_obj4.predict() - X_test
        rmse4 = np.sqrt(np.mean(err4 ** 2))

        fit_obj5.fit(X_train)
        err5 = fit_obj5.predict() - X_test
        rmse5 = np.sqrt(np.mean(err5 ** 2))

        fit_obj6.fit(X_train)
        preds = fit_obj6.predict(return_std=True)

        fit_obj7.fit(X_train)
        preds2 = fit_obj7.predict(return_std=True)

        fit_obj8.fit(X_train)
        preds3 = fit_obj8.predict(return_std=True)

        fit_obj9.fit(X_train)
        preds4 = fit_obj9.predict(return_std=True)

        fit_obj9.fit(X_train[:, 0])
        preds5 = fit_obj9.predict(return_std=True)

        fit_obj10.fit(X_train)
        preds6 = fit_obj10.predict(return_std=True)

        
        self.assertTrue(np.allclose(rmse, 10.395403649098926))
        self.assertTrue(np.allclose(rmse_xreg, 10.401634671488587))
        self.assertTrue(np.allclose(rmse2, 10.395285391773202))        
        self.assertTrue(np.allclose(rmse3, 10.394290838542101))        
        self.assertTrue(np.allclose(rmse4, 10.371173921434293))        
        self.assertTrue(np.allclose(rmse5, 10.402884770399375, atol=1e-3))

        self.assertTrue(np.allclose(preds[1][0], 0.4187793))
        self.assertTrue(np.allclose(preds2[1][0], 0.05719732))
        self.assertTrue(np.allclose(preds6[1][0], 0.06910578))
        self.assertTrue(np.allclose(preds4[1][0], 0.06910578))
        self.assertTrue(np.allclose(preds5[1][0], 0.0759192))
        self.assertTrue(np.allclose(preds3[1][0], 0.07400078))

    def test_get_set(self):

        np.random.seed(123)
        X = np.random.rand(25, 3)
        X[:, 0] = 100 * X[:, 0]
        X[:, 2] = 25 * X[:, 2]

        regr = linear_model.BayesianRidge()

        fit_obj = ns.MTS(
            regr,
            n_hidden_features=10,
            direct_link=False,
            bias=False,
            nodes_sim="sobol",
            type_scaling=("std", "minmax", "std"),
            activation_name="relu",
            n_clusters=0,
        )

        fit_obj.set_params(
            n_hidden_features=5,
            activation_name="relu",
            a=0.01,
            nodes_sim="sobol",
            bias=True,
            direct_link=True,
            n_clusters=None,
            type_clust="kmeans",
            type_scaling=("std", "std", "std"),
            seed=123,
            lags=1,
        )

        fit_obj2 = ns.MTS(
            regr,
            n_hidden_features=10,
            direct_link=False,
            bias=False,
            dropout=0.5,
            nodes_sim="sobol",
            type_scaling=("std", "minmax", "std"),
            activation_name="relu",
            n_clusters=0,
        )

        self.assertTrue(
            (fit_obj.get_params()["lags"] == 1)
            & (fit_obj.get_params()["type_scaling"] == ("std", "std", "std"))
            & (fit_obj2.get_params()["obj__lambda_1"] == 1e-06)
        )

    def test_score(self):

        np.random.seed(123)
        X = np.random.rand(25, 3)
        X[:, 0] = 100 * X[:, 0]
        X[:, 2] = 25 * X[:, 2]

        regr = linear_model.BayesianRidge()

        fit_obj = ns.MTS(
            regr,
            n_hidden_features=10,
            direct_link=False,
            bias=False,
            nodes_sim="sobol",
            type_scaling=("std", "minmax", "std"),
            activation_name="relu",
            n_clusters=0,
        )

        scores = fit_obj.score(
            X,
            training_index=range(20),
            testing_index=range(20, 25),
            scoring="neg_mean_squared_error",
        )

        scores2 = fit_obj.score(
            X, training_index=range(20), testing_index=range(20, 25)
        )

        # self.assertTrue(
        #     np.allclose([np.round(x) for x in scores], [239.0, 0.0, 85.0])
        #     & np.allclose(
        #         [np.round(x, 3) for x in scores2], [15.464, 0.284, 9.22]
        #     )
        # )


if __name__ == "__main__":
    ut.main()
