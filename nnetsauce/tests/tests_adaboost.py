import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import unittest as ut
import nnetsauce as ns
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer, load_wine


class TestAdaBoost(ut.TestCase):
    def test_AdaBoost(self):

        breast_cancer = load_breast_cancer()
        X = breast_cancer.data
        y = breast_cancer.target
        np.random.seed(123)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=123
        )

        wine = load_wine()
        Z = wine.data
        t = wine.target
        Z_train, Z_test, t_train, t_test = train_test_split(
            Z, t, test_size=0.2, random_state=123
        )

        clf = DecisionTreeClassifier(
            max_depth=1, random_state=123
        )
        fit_obj = ns.AdaBoostClassifier(
            clf,
            n_hidden_features=np.int(11.22338867),
            direct_link=True,
            n_estimators=5,
            learning_rate=0.01126343,
            col_sample=0.72684326,
            row_sample=0.86429443,
            dropout=0.63078613,
            n_clusters=0,
            verbose=0,
            seed=123,
            reg_lambda=0,
            reg_alpha=0,
            method="SAMME.R",
        )

        clf2 = DecisionTreeClassifier(
            max_depth=1, random_state=123
        )
        fit_obj2 = ns.AdaBoostClassifier(
            clf2,
            n_hidden_features=np.int(8.21154785e01),
            direct_link=True,
            n_estimators=5,
            learning_rate=2.96252441e-02,
            col_sample=4.22766113e-01,
            row_sample=7.87268066e-01,
            dropout=1.56909180e-01,
            n_clusters=0,
            verbose=0,
            seed=123,
            reg_lambda=0,
            reg_alpha=0,
            method="SAMME",
        )

        clf3 = DecisionTreeClassifier(
            max_depth=1, random_state=123
        )
        fit_obj3 = ns.AdaBoostClassifier(
            clf3,
            n_hidden_features=np.int(11.22338867),
            direct_link=True,
            n_estimators=5,
            learning_rate=0.01126343,
            col_sample=0.72684326,
            row_sample=0.86429443,
            dropout=0.63078613,
            n_clusters=0,
            verbose=0,
            seed=123,
            reg_lambda=0.1,
            reg_alpha=0.5,
            method="SAMME.R",
        )

        clf4 = DecisionTreeClassifier(
            max_depth=1, random_state=123
        )
        fit_obj4 = ns.AdaBoostClassifier(
            clf4,
            n_hidden_features=np.int(11.22338867),
            direct_link=True,
            n_estimators=5,
            learning_rate=0.01126343,
            col_sample=0.72684326,
            row_sample=0.86429443,
            dropout=0.63078613,
            n_clusters=2,
            cluster_encode=False,
            verbose=0,
            seed=123,
            reg_lambda=0.1,
            reg_alpha=0.5,
            method="SAMME.R",
        )

        fit_obj.fit(X_train, y_train)
        preds1 = fit_obj.predict_proba(X_test)

        fit_obj2.fit(Z_train, t_train)
        preds2 = fit_obj2.predict_proba(Z_test)

        fit_obj3.fit(Z_train, t_train)
        preds3 = fit_obj3.predict_proba(Z_test)

        fit_obj4.fit(Z_train, t_train)
        preds4 = fit_obj4.predict_proba(Z_test)

        self.assertTrue(
            np.allclose(preds1[0, 0], 0.0010398157809255607)
        )

        self.assertTrue(
            np.allclose(preds1[0, 1], 0.9989601842190745)
        )

        self.assertTrue(
            np.allclose(preds2[0, 0], 0.28646471034732585)
        )

        self.assertFalse(np.allclose(preds3[0, 0], 1000))

        self.assertFalse(np.allclose(preds4[0, 0], 1000))

        self.assertTrue(
            np.allclose(fit_obj.predict(X_test)[0], 1)
            & np.allclose(fit_obj2.predict(X_test)[0], 1)
        )

    def test_score(self):

        breast_cancer = load_breast_cancer()
        X = breast_cancer.data
        y = breast_cancer.target

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=123
        )

        clf = LogisticRegression(
            solver="liblinear", multi_class="ovr"
        )
        fit_obj = ns.AdaBoostClassifier(
            clf,
            n_hidden_features=np.int(11.22338867),
            direct_link=True,
            n_estimators=5,
            learning_rate=0.01126343,
            col_sample=0.72684326,
            row_sample=0.86429443,
            dropout=0.63078613,
            n_clusters=0,
            verbose=0,
            seed=123,
            reg_lambda=0,
            reg_alpha=0,
            method="SAMME.R",
        )

        clf2 = LogisticRegression(
            solver="liblinear", multi_class="ovr"
        )
        fit_obj2 = ns.AdaBoostClassifier(
            clf2,
            n_hidden_features=np.int(8.21154785e01),
            direct_link=True,
            n_estimators=5,
            learning_rate=2.96252441e-02,
            col_sample=4.22766113e-01,
            row_sample=7.87268066e-01,
            dropout=1.56909180e-01,
            n_clusters=0,
            verbose=0,
            seed=123,
            reg_lambda=0,
            reg_alpha=0,
            method="SAMME",
        )

        fit_obj3 = ns.AdaBoostClassifier(
            clf,
            n_hidden_features=np.int(11.22338867),
            direct_link=True,
            n_estimators=5,
            learning_rate=0.01126343,
            col_sample=0.72684326,
            row_sample=0.86429443,
            dropout=0.63078613,
            n_clusters=0,
            verbose=0,
            seed=123,
            reg_lambda=0.1,
            reg_alpha=0.5,
            method="SAMME.R",
        )

        fit_obj.fit(X_train, y_train)
        score1 = fit_obj.score(X_test, y_test)

        fit_obj2.fit(X_train, y_train)
        score2 = fit_obj2.score(X_test, y_test)

        fit_obj3.fit(X_train, y_train)
        score3 = fit_obj3.score(X_test, y_test)

        self.assertTrue(
            np.allclose(score1, 0.9210526315789473)
            & np.allclose(score2, 0.7807017543859649)
        )

        self.assertFalse(np.allclose(score3, 1000))


if __name__ == "__main__":
    ut.main()
