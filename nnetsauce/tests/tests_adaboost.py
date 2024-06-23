import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import unittest as ut
import nnetsauce as ns
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer, load_wine

print(f"\n ----- Running: {os.path.basename(__file__)}... ----- \n")


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

        clf = DecisionTreeClassifier(max_depth=1, random_state=123)
        fit_obj = ns.AdaBoostClassifier(
            clf,
            n_hidden_features=np.int32(11.22338867),
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

        clf2 = DecisionTreeClassifier(max_depth=1, random_state=123)
        fit_obj2 = ns.AdaBoostClassifier(
            clf2,
            n_hidden_features=np.int32(8.21154785e01),
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

        clf3 = DecisionTreeClassifier(max_depth=1, random_state=123)
        fit_obj3 = ns.AdaBoostClassifier(
            clf3,
            n_hidden_features=np.int32(11.22338867),
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

        clf4 = DecisionTreeClassifier(max_depth=1, random_state=123)
        fit_obj4 = ns.AdaBoostClassifier(
            clf4,
            n_hidden_features=np.int32(11.22338867),
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

        clf5 = DecisionTreeClassifier(max_depth=1, random_state=123)
        fit_obj5 = ns.AdaBoostClassifier(
            clf5,
            n_hidden_features=np.int32(11.22338867),
            direct_link=True,
            n_estimators=5,
            learning_rate=0.01126343,
            col_sample=0.72684326,
            row_sample=0.86429443,
            dropout=0.63078613,
            n_clusters=2,
            cluster_encode=False,
            verbose=1,
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

        fit_obj5.fit(
            Z_train,
            t_train,
            sample_weight=np.repeat(1.0 / len(t_train), len(t_train)),
        )
        preds5 = fit_obj5.predict_proba(Z_test)

        print(preds1[0, 0])
        print(preds2[0, 0])
        print(preds3[0, 0])
        print(preds4[0, 0])
        print(preds5[0, 0])

        self.assertTrue(np.allclose(preds1[0, 0], 0.0016485697773120423))

        self.assertTrue(np.allclose(preds2[0, 0], 0.2874144965263139))

        self.assertTrue(np.allclose(preds3[0, 0], 1.4485399945960407e-64))

        self.assertTrue(np.allclose(preds4[0, 0], 3.041752189425878e-42))

        self.assertTrue(np.allclose(preds5[0, 0], 3.041752189425878e-42))

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

        clf = LogisticRegression(solver="liblinear", multi_class="ovr")
        fit_obj = ns.AdaBoostClassifier(
            clf,
            n_hidden_features=np.int32(11.22338867),
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

        clf2 = LogisticRegression(solver="liblinear", multi_class="ovr")
        fit_obj2 = ns.AdaBoostClassifier(
            clf2,
            n_hidden_features=np.int32(8.21154785e01),
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
            n_hidden_features=np.int32(11.22338867),
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
        probs1 = fit_obj.predict_proba(X_test)
        preds1 = fit_obj.predict(X_test)

        fit_obj2.fit(X_train, y_train)
        score2 = fit_obj2.score(X_test, y_test)
        probs2 = fit_obj2.predict_proba(X_test)
        preds2 = fit_obj2.predict(X_test)

        fit_obj3.fit(X_train, y_train)
        score3 = fit_obj3.score(X_test, y_test)
        probs3 = fit_obj3.predict_proba(X_test)
        preds3 = fit_obj3.predict(X_test)

        print(f"score1: {score1}")
        print(f"score2: {score2}")
        print(f"score3: {score3}")

        self.assertTrue(
            np.allclose(score1, 0.9210526315789473)
            & np.allclose(score2, 0.8157894736842105)
            & np.allclose(score3, 0.9210526315789473)
        )

        print(f"probs1: {probs1}")
        print(f"probs2: {probs2}")
        print(f"probs3: {probs3}")

        self.assertTrue(
            np.allclose(probs1[0, 0], 2.90334824e-19)
            & np.allclose(probs2[0, 0], 0.40325828)
            & np.allclose(probs3[0, 0], 2.90334824e-19)
        )

        print(f"preds1: {preds1}")
        print(f"preds2: {preds2}")
        print(f"preds3: {preds3}")

        self.assertTrue(
            np.allclose(preds1[0], 1)
            & np.allclose(preds2[0], 1)
            & np.allclose(preds3[0], 1)
        )


if __name__ == "__main__":
    ut.main()
