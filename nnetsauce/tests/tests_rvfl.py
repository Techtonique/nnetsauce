import numpy as np
from sklearn import datasets
import unittest as ut
import nnetsauce as ns


# Basic tests


class TestRVFL(ut.TestCase):
    def test_rvfl(self):

        np.random.seed(123)
        X, y = datasets.make_regression(n_samples=25, n_features=3)

        fit_obj = ns.BayesianRVFLRegressor(
            n_hidden_features=10,
            direct_link=False,
            bias=False,
            nodes_sim="sobol",
            type_scaling=("std", "minmax", "std"),
            activation_name="relu",
            n_clusters=0,
        )

        fit_obj2 = ns.BayesianRVFL2Regressor(
            n_hidden_features=9,
            direct_link=False,
            bias=True,
            nodes_sim="halton",
            type_scaling=("std", "minmax", "minmax"),
            activation_name="sigmoid",
            n_clusters=2,
        )

        fit_obj3 = ns.BayesianRVFLRegressor(
            n_hidden_features=8,
            direct_link=True,
            bias=False,
            nodes_sim="uniform",
            type_scaling=("minmax", "minmax", "std"),
            activation_name="tanh",
            n_clusters=3,
        )

        fit_obj4 = ns.BayesianRVFL2Regressor(
            n_hidden_features=7,
            direct_link=True,
            bias=True,
            nodes_sim="hammersley",
            type_scaling=("minmax", "minmax", "minmax"),
            activation_name="elu",
            n_clusters=4,
        )

        fit_obj5 = ns.BayesianRVFL2Regressor(
            n_hidden_features=7,
            direct_link=True,
            bias=True,
            nodes_sim="hammersley",
            type_scaling=("minmax", "minmax", "minmax"),
            activation_name="elu",
            n_clusters=4,
            cluster_encode=True,
        )

        index_train = range(20)
        index_test = range(20, 25)
        X_train = X[index_train, :]
        y_train = y[index_train]
        X_test = X[index_test, :]
        y_test = y[index_test]

        fit_obj.fit(X_train, y_train)
        err = fit_obj.predict(X_test, return_std=True)[0] - y_test
        rmse = np.sqrt(np.mean(err ** 2))

        fit_obj2.fit(X_train, y_train)
        err2 = fit_obj2.predict(X_test, return_std=True)[0] - y_test
        rmse2 = np.sqrt(np.mean(err2 ** 2))

        fit_obj3.fit(X_train, y_train)
        err3 = fit_obj3.predict(X_test, return_std=True)[0] - y_test
        rmse3 = np.sqrt(np.mean(err3 ** 2))

        fit_obj4.fit(X_train, y_train)
        err4 = fit_obj4.predict(X_test, return_std=True)[0] - y_test
        rmse4 = np.sqrt(np.mean(err4 ** 2))

        fit_obj5.fit(X_train, y_train)
        err5 = fit_obj5.predict(X_test, return_std=True)[0] - y_test
        rmse5 = np.sqrt(np.mean(err5 ** 2))

        pred1 = fit_obj.predict(X_test[0, :], return_std=True)[0]

        pred2 = fit_obj2.predict(X_test[0, :], return_std=True)[0]

        pred3 = fit_obj.predict(X_test[0, :], return_std=False)

        pred4 = fit_obj4.predict(X_test[0, :], return_std=False)

        self.assertTrue(
            np.allclose(rmse, 0.81893186154747988)
            & np.allclose(rmse2, 17.278090150613096)
            & np.allclose(rmse3, 32.582378601766486)
            & np.allclose(rmse4, 53.052553013478331)
            & np.allclose(rmse5, 53.052553013478295)
            & np.allclose(pred1, 325.96545701774187)
            & np.allclose(pred2, 299.56243221494879)
            & np.allclose(pred3, 325.96545701774187)
            & np.allclose(pred4, 234.76784640807193)
        )

    def test_get_set(self):

        fit_obj = ns.BayesianRVFLRegressor(
            n_hidden_features=10,
            direct_link=False,
            bias=False,
            nodes_sim="sobol",
            type_scaling=("std", "minmax", "std"),
            activation_name="relu",
            n_clusters=0,
        )

        fit_obj2 = ns.BayesianRVFL2Regressor(
            n_hidden_features=9,
            direct_link=False,
            bias=True,
            nodes_sim="halton",
            type_scaling=("std", "minmax", "minmax"),
            activation_name="sigmoid",
            n_clusters=2,
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
            type_scaling=("std", "minmax", "std"),
            seed=123,
            s=0.1,
            sigma=0.05,
            return_std=True,
        )

        fit_obj2.set_params(
            n_hidden_features=4,
            activation_name="relu",
            a=0.01,
            nodes_sim="sobol",
            bias=True,
            dropout=0.5,
            direct_link=True,
            n_clusters=None,  # optim
            type_clust="kmeans",
            type_scaling=("std", "std", "minmax"),
            seed=123,
            s1=0.1,
            s2=0.1,
            sigma=0.05,  # optim
            return_std=True,
        )

    def test_score(self):

        np.random.seed(123)
        X, y = datasets.make_regression(n_samples=100, n_features=3)

        fit_obj = ns.BayesianRVFLRegressor(
            n_hidden_features=10,
            direct_link=False,
            bias=False,
            nodes_sim="sobol",
            type_scaling=("std", "minmax", "std"),
            activation_name="relu",
            n_clusters=0,
        )

        fit_obj2 = ns.BayesianRVFL2Regressor(
            n_hidden_features=9,
            direct_link=False,
            bias=True,
            nodes_sim="halton",
            type_scaling=("std", "minmax", "minmax"),
            activation_name="sigmoid",
            n_clusters=2,
        )

        fit_obj3 = ns.BayesianRVFL2Regressor(
            n_hidden_features=9,
            direct_link=True,
            bias=True,
            dropout=0.3,
            nodes_sim="halton",
            type_scaling=("std", "minmax", "minmax"),
            activation_name="sigmoid",
            n_clusters=2,
        )

        fit_obj4 = ns.BayesianRVFL2Regressor(
            n_hidden_features=9,
            direct_link=True,
            bias=True,
            dropout=0.5,
            nodes_sim="halton",
            type_scaling=("std", "minmax", "minmax"),
            activation_name="sigmoid",
            n_clusters=2,
        )

        fit_obj.fit(X, y)
        fit_obj2.fit(X, y)
        fit_obj3.fit(X, y)
        fit_obj4.fit(X, y)
        fit_obj4.set_params(return_std=True)

        self.assertTrue(
            np.allclose(
                fit_obj.score(X, y, scoring="neg_mean_squared_error"),
                0.023104115093245361,
            )
            & np.allclose(fit_obj.score(X, y), 0.023104115093245361)
            & np.allclose(
                fit_obj2.score(X, y, scoring="neg_mean_squared_error"),
                51.485414634058536,
            )
            & np.allclose(fit_obj2.score(X, y), 51.485414634058536)
            & np.allclose(fit_obj3.score(X, y), 0.20023262498412012)
            & np.allclose(fit_obj4.score(X, y), 0.17517631177933579)
        )


if __name__ == "__main__":
    ut.main()
