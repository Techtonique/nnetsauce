import numpy as np
from sklearn import datasets, linear_model, gaussian_process
import unittest as ut
import nnetsauce as ns
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer, load_wine
from sklearn.linear_model import ElasticNet, LinearRegression


class TestMultitask(ut.TestCase):
    def test_Multitask(self):

        breast_cancer = load_breast_cancer()
        X = breast_cancer.data
        y = breast_cancer.target
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=71023
        )

        wine = load_wine()
        Z = wine.data
        t = wine.target
        Z_train, Z_test, t_train, t_test = train_test_split(
            Z, t, test_size=0.2, random_state=61283
        )

        regr = LinearRegression()

        fit_obj = ns.MultitaskClassifier(
            regr, n_hidden_features=5, n_clusters=0
        )

        fit_obj2 = ns.MultitaskClassifier(
            regr, n_hidden_features=10, n_clusters=3
        )

        fit_obj3 = ns.MultitaskClassifier(
            regr, n_hidden_features=5, n_clusters=0
        )

        fit_obj4 = ns.MultitaskClassifier(
            regr, n_hidden_features=8, n_clusters=2
        )

        fit_obj.fit(X_train, y_train)
        preds1 = fit_obj.predict_proba(X_test)

        fit_obj2.fit(X_train, y_train)
        preds2 = fit_obj.predict_proba(X_test)

        fit_obj3.fit(Z_train, t_train)
        preds3 = fit_obj3.predict_proba(Z_test)

        fit_obj4.fit(Z_train, t_train)
        preds4 = fit_obj4.predict_proba(Z_test)

        self.assertTrue(np.allclose(preds1[0, 0], 0.9999866746110515))
        self.assertTrue(np.allclose(preds2[0, 0], 0.9999866746110515))
        self.assertTrue(np.allclose(preds3[0, 0], 0.07875978450669445))
        self.assertTrue(np.allclose(preds4[0, 0], 0.3944006512204142))

        self.assertTrue(
            np.allclose(fit_obj.predict(X_test)[0], 0)
            & np.allclose(fit_obj2.predict(X_test)[0], 1)
            & np.allclose(fit_obj3.predict(Z_test)[0], 1)
            & np.allclose(fit_obj4.predict(Z_test)[0], 0)
        )

    def test_score(self):
        breast_cancer = load_breast_cancer()
        X = breast_cancer.data
        y = breast_cancer.target
        np.random.seed(123)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        wine = load_wine()
        Z = wine.data
        t = wine.target
        np.random.seed(123)
        Z_train, Z_test, t_train, t_test = train_test_split(Z, t, test_size=0.2)

        regr = LinearRegression()

        fit_obj = ns.MultitaskClassifier(
            regr, n_hidden_features=3, n_clusters=2
        )

        fit_obj2 = ns.MultitaskClassifier(
            regr, n_hidden_features=7, n_clusters=2
        )

        fit_obj3 = ns.MultitaskClassifier(
            regr, n_hidden_features=5, n_clusters=2
        )

        fit_obj4 = ns.MultitaskClassifier(
            regr, n_hidden_features=4, n_clusters=2
        )

        fit_obj.fit(X_train, y_train)
        score1 = fit_obj.score(X_test, y_test)

        fit_obj2.fit(X_train, y_train)
        score2 = fit_obj2.score(X_test, y_test)

        fit_obj3.fit(Z_train, t_train)
        score3 = fit_obj3.score(Z_test, t_test)

        fit_obj4.fit(Z_train, t_train)
        score4 = fit_obj4.score(Z_test, t_test)

        self.assertTrue(np.allclose(score1, 0.956140350877193))

        self.assertTrue(np.allclose(score2, 0.9649122807017544))

        self.assertTrue(np.allclose(score3, 0.9722222222222222))

        self.assertTrue(np.allclose(score4, 0.9722222222222222))


if __name__ == "__main__":
    ut.main()
