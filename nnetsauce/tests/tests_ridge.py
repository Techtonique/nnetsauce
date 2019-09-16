import numpy as np
from sklearn import datasets, linear_model, gaussian_process
import unittest as ut
import nnetsauce as ns
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer, load_wine


class TestRidge(ut.TestCase):
    def test_Ridge(self):

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

        fit_obj = ns.RidgeClassifier(
            lambda1=0.025,
            lambda2=0.5,
            n_hidden_features=5,
            n_clusters=0,
        )

        fit_obj2 = ns.RidgeClassifier(
            lambda1=0.01,
            lambda2=0.01,
            n_hidden_features=10,
            n_clusters=2,
        )

        fit_obj3 = ns.RidgeClassifier(
            lambda1=0.025,
            lambda2=0.05,
            n_hidden_features=5,
            n_clusters=0,
        )

        fit_obj4 = ns.RidgeClassifier(
            lambda1=0.001,
            lambda2=0.01,
            n_hidden_features=10,
            n_clusters=2,
        )

        fit_obj.fit(X_train, y_train)
        preds1 = fit_obj.predict_proba(X_test)

        fit_obj2.fit(X_train, y_train)
        preds2 = fit_obj.predict_proba(X_test)

        fit_obj3.fit(Z_train, t_train)
        preds3 = fit_obj3.predict_proba(Z_test)

        fit_obj4.fit(Z_train, t_train)
        preds4 = fit_obj4.predict_proba(Z_test)

        self.assertTrue(
            np.allclose(
                preds1[0, 0], 5.1412488698250085e-06
            )
        )
        self.assertTrue(
            np.allclose(
                preds2[0, 0], 5.1412488698250085e-06
            )
        )
        self.assertTrue(
            np.allclose(preds3[0, 0], 0.0488412175)
        )
        self.assertTrue(
            np.allclose(preds4[0, 0], 0.8545733738)
        )

        self.assertTrue(
            np.allclose(fit_obj.predict(X_test)[0], 1)
            & np.allclose(fit_obj2.predict(X_test)[0], 1)
            & np.allclose(fit_obj3.predict(Z_test)[0], 1)
            & np.allclose(fit_obj4.predict(Z_test)[0], 0)
        )

    def test_score(self):
        breast_cancer = load_breast_cancer()
        X = breast_cancer.data
        y = breast_cancer.target
        np.random.seed(123)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2
        )

        wine = load_wine()
        Z = wine.data
        t = wine.target
        np.random.seed(123)
        Z_train, Z_test, t_train, t_test = train_test_split(
            Z, t, test_size=0.2
        )

        fit_obj = ns.RidgeClassifier(
            lambda1=0.025,
            lambda2=0.5,
            n_hidden_features=5,
            n_clusters=2,
        )

        fit_obj2 = ns.RidgeClassifier(
            lambda1=0.01,
            lambda2=0.01,
            n_hidden_features=10,
            n_clusters=2,
        )

        fit_obj3 = ns.RidgeClassifier(
            lambda1=0.025,
            lambda2=0.05,
            n_hidden_features=5,
            n_clusters=2,
        )

        fit_obj4 = ns.RidgeClassifier(
            lambda1=0.001,
            lambda2=0.01,
            n_hidden_features=10,
            n_clusters=2,
        )

        fit_obj.fit(X_train, y_train)
        score1 = fit_obj.score(X_test, y_test)

        fit_obj2.fit(X_train, y_train)
        score2 = fit_obj2.score(X_test, y_test)

        fit_obj3.fit(Z_train, t_train)
        score3 = fit_obj3.score(Z_test, t_test)

        fit_obj4.fit(Z_train, t_train)
        score4 = fit_obj4.score(Z_test, t_test)

        self.assertTrue(
            np.allclose(score1, 0.9649122807017544)
        )

        self.assertTrue(
            np.allclose(score2, 0.94736842105263153)
        )

        self.assertTrue(np.allclose(score3, 1.0))

        self.assertTrue(
            np.allclose(score4, 0.9722222222222222)
        )


if __name__ == "__main__":
    ut.main()
