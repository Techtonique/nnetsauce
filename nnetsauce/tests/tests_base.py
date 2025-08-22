import os
import numpy as np
import unittest as ut
import nnetsauce as ns

from sklearn import datasets
from .utilstests import test_check

print(f"\n ----- Running: {os.path.basename(__file__)}... ----- \n")

np.random.seed(123)


class TestBase(ut.TestCase):
    def test_base(self):
        X, y = datasets.make_regression(
            n_samples=25, n_features=3, random_state=123
        )

        fit_obj = ns.BaseRegressor(
            n_hidden_features=10,
            direct_link=False,
            bias=False,
            nodes_sim="sobol",
            activation_name="relu",
            n_clusters=0,
        )

        fit_obj.set_params(
            n_hidden_features=5,
            activation_name="relu",
            a=0.01,
            nodes_sim="sobol",
            bias=False,
            direct_link=False,
            n_clusters=2,
            type_clust="gmm",
            type_scaling=("std", "std", "minmax"),
        )

        fit_obj2 = ns.BaseRegressor(
            n_hidden_features=9,
            direct_link=False,
            bias=True,
            nodes_sim="halton",
            activation_name="sigmoid",
            n_clusters=2,
        )

        fit_obj3 = ns.BaseRegressor(
            n_hidden_features=8,
            direct_link=True,
            bias=False,
            nodes_sim="uniform",
            activation_name="tanh",
            n_clusters=3,
        )

        fit_obj4 = ns.BaseRegressor(
            n_hidden_features=7,
            direct_link=True,
            bias=True,
            nodes_sim="hammersley",
            activation_name="elu",
            n_clusters=4,
        )

        fit_obj5 = ns.BaseRegressor(
            n_hidden_features=2,
            direct_link=True,
            bias=True,
            nodes_sim="hammersley",
            activation_name="prelu",
            n_clusters=0,
        )

        fit_obj6 = ns.BaseRegressor(
            n_hidden_features=2,
            direct_link=True,
            bias=True,
            nodes_sim="hammersley",
            activation_name="prelu",
            n_clusters=0,
        )

        fit_obj6.set_params(
            n_hidden_features=5,
            activation_name="elu",
            a=0.01,
            nodes_sim="sobol",
            bias=False,
            direct_link=False,
            n_clusters=2,
            type_clust="gmm",
            type_scaling=("std", "std", "minmax"),
        )

        fit_obj7 = ns.BaseRegressor(
            n_hidden_features=2,
            direct_link=True,
            bias=True,
            nodes_sim="hammersley",
            activation_name="elu",
            n_clusters=0,
        )

        fit_obj7.set_params(
            n_hidden_features=5,
            activation_name="prelu",
            a=0.01,
            nodes_sim="sobol",
            bias=False,
            direct_link=False,
            n_clusters=2,
            type_clust="gmm",
            type_scaling=("std", "std", "minmax"),
        )

        fit_obj8 = ns.BaseRegressor(
            n_hidden_features=2,
            direct_link=True,
            bias=False,
            nodes_sim="hammersley",
            activation_name="elu",
            n_clusters=0,
        )

        fit_obj9 = ns.BaseRegressor(
            n_hidden_features=2,
            direct_link=True,
            bias=False,
            nodes_sim="halton",
            activation_name="elu",
            n_clusters=0,
        )

        fit_obj10 = ns.BaseRegressor(
            n_hidden_features=3,
            direct_link=True,
            bias=True,
            nodes_sim="uniform",
            activation_name="tanh",
            n_clusters=3,
            seed=5610,
        )

        fit_obj11 = ns.BaseRegressor(
            n_hidden_features=3,
            direct_link=True,
            bias=True,
            nodes_sim="uniform",
            activation_name="tanh",
            n_clusters=0,
            col_sample=1,
            seed=5260,
        )

        fit_obj12 = ns.BaseRegressor(
            n_hidden_features=3,
            direct_link=True,
            bias=True,
            nodes_sim="uniform",
            activation_name="tanh",
            n_clusters=0,
            col_sample=0.8,
            seed=2763,
        )

        fit_obj13 = ns.BaseRegressor(
            n_hidden_features=3,
            direct_link=True,
            bias=True,
            nodes_sim="uniform",
            activation_name="tanh",
            n_clusters=2,
            cluster_encode=False,
            col_sample=0.8,
            seed=2763,
        )

        index_train = range(20)
        index_test = range(20, 25)
        X_train = X[index_train, :]
        y_train = y[index_train]
        X_test = X[index_test, :]
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

        fit_obj5.fit(X_train, y_train)
        err5 = fit_obj5.predict(X_test) - y_test
        rmse5 = np.sqrt(np.mean(err5**2))

        fit_obj6.fit(X_train, y_train)
        err6 = fit_obj6.predict(X_test) - y_test
        rmse6 = np.sqrt(np.mean(err6**2))

        fit_obj7.fit(X_train, y_train)
        err7 = fit_obj7.predict(X_test) - y_test
        rmse7 = np.sqrt(np.mean(err7**2))

        fit_obj8.fit(X_train, y_train)
        err8 = fit_obj8.predict(X_test) - y_test
        rmse8 = np.sqrt(np.mean(err8**2))

        fit_obj9.fit(X_train, y_train)
        err9 = fit_obj9.predict(X_test) - y_test
        rmse9 = np.sqrt(np.mean(err9**2))

        fit_obj10.fit(X_train, y_train)
        err10 = fit_obj10.predict(X_test) - y_test
        rmse10 = np.sqrt(np.mean(err10**2))

        fit_obj11.fit(X_train, y_train)
        err11 = fit_obj11.predict(X_test) - y_test
        rmse11 = np.sqrt(np.mean(err11**2))

        fit_obj12.fit(X_train, y_train)
        err12 = fit_obj12.predict(X_test) - y_test
        rmse12 = np.sqrt(np.mean(err12**2))

        fit_obj13.fit(X_train, y_train)
        err13 = fit_obj13.predict(X_test) - y_test
        rmse13 = np.sqrt(np.mean(err13**2))

        print(f"rmse: {rmse}")
        print(f"rmse2: {rmse2}")
        print(f"rmse3: {rmse3}")
        print(f"rmse4: {rmse4}")
        print(f"rmse5: {rmse5}")
        print(f"rmse6: {rmse6}")
        print(f"rmse7: {rmse7}")
        print(f"rmse8: {rmse8}")
        print(f"rmse9: {rmse9}")
        print(f"rmse10: {rmse10}")
        print(f"rmse11: {rmse11}")
        print(f"rmse12: {rmse12}")
        print(f"rmse13: {rmse13}")

        self.assertTrue(test_check(rmse, 63.24))
        self.assertTrue(test_check(rmse2, 19.40))
        self.assertFalse(test_check(rmse3, 78.94))
        self.assertTrue(test_check(rmse4, 195.92))
        self.assertTrue(test_check(rmse5, 22.54))
        self.assertTrue(test_check(rmse6, 62.61))
        self.assertTrue(test_check(rmse7, 63.05))
        self.assertTrue(test_check(rmse8, 22.54))
        self.assertTrue(test_check(rmse9, 22.54))
        self.assertTrue(test_check(rmse10, 2299905125015368))
        self.assertTrue(test_check(rmse11, 22.54))
        self.assertTrue(test_check(rmse12, 22.54))
        self.assertTrue(test_check(rmse13, 1e-12))

    def test_score(self):
        X, y = datasets.make_regression(
            n_samples=25, n_features=3, random_state=123
        )

        fit_obj = ns.BaseRegressor(
            n_hidden_features=5,
            direct_link=True,
            bias=True,
            nodes_sim="sobol",
            activation_name="relu",
            n_clusters=2,
        )
        fit_obj.fit(X, y)

        r2 = fit_obj.score(X, y, scoring="r2")
        rmse = fit_obj.score(X, y)
        print(f"r2: {r2}")
        print(f"rmse: {rmse}")

        self.assertTrue(test_check(r2, 0.93) & test_check(rmse, 39.95))


if __name__ == "__main__":
    ut.main()
