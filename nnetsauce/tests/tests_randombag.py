import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
import unittest as ut
import nnetsauce as ns
from sklearn.model_selection import train_test_split
from sklearn.datasets import (
    load_breast_cancer,
    load_wine,
    fetch_california_housing,
)
from sklearn.linear_model import LogisticRegression


class TestRandomBag(ut.TestCase):
    def test_RandomBag(self):

        breast_cancer = load_breast_cancer()
        X = breast_cancer.data
        y = breast_cancer.target
        np.random.seed(123)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=19823
        )

        wine = load_wine()
        Z = wine.data
        t = wine.target
        Z_train, Z_test, t_train, t_test = train_test_split(
            Z, t, test_size=0.2, random_state=12736
        )

        X2, y2 = fetch_california_housing(return_X_y=True, as_frame=False)
        # split data into training test and test set
        X_train2, X_test2, y_train2, y_test2 = train_test_split(
            X2, y2, test_size=0.2, random_state=13
        )

        clf = DecisionTreeClassifier(max_depth=1, random_state=14235)
        fit_obj = ns.RandomBagClassifier(
            clf,
            n_estimators=3,
            n_hidden_features=2,
            activation_name="relu",
            a=0.01,
            nodes_sim="sobol",
            bias=True,
            dropout=0,
            direct_link=False,
            n_clusters=2,
            type_clust="kmeans",
            type_scaling=("std", "std", "std"),
            col_sample=0.9,
            row_sample=0.9,
            n_jobs=None,
            seed=14253,
            verbose=0,
        )

        clf2 = DecisionTreeClassifier(max_depth=1, random_state=15243)
        fit_obj2 = ns.RandomBagClassifier(
            clf2,
            n_estimators=3,
            n_hidden_features=2,
            activation_name="relu",
            a=0.01,
            nodes_sim="sobol",
            bias=True,
            dropout=0,
            direct_link=False,
            n_clusters=2,
            type_clust="kmeans",
            type_scaling=("std", "std", "std"),
            col_sample=0.9,
            row_sample=0.9,
            n_jobs=None,
            seed=19237,
            verbose=0,
        )

        clf3 = DecisionTreeClassifier(max_depth=1, random_state=15243)
        fit_obj3 = ns.RandomBagClassifier(
            clf3,
            n_estimators=3,
            n_hidden_features=2,
            activation_name="relu",
            a=0.01,
            nodes_sim="sobol",
            bias=True,
            dropout=0,
            direct_link=False,
            n_clusters=2,
            cluster_encode=False,
            type_clust="kmeans",
            type_scaling=("std", "std", "std"),
            col_sample=0.9,
            row_sample=0.9,
            n_jobs=None,
            seed=19237,
            verbose=0,
        )

        obj = DecisionTreeRegressor(max_depth=3, random_state=123)
        fit_obj4 = ns.RandomBagRegressor(
            obj=obj,
            direct_link=False,
            n_estimators=50,
            col_sample=0.9,
            row_sample=0.9,
            dropout=0,
            n_clusters=0,
            verbose=1,
        )

        fit_obj.fit(X_train, y_train)
        preds1 = fit_obj.predict_proba(X_test)

        fit_obj2.fit(Z_train, t_train)
        preds2 = fit_obj2.predict_proba(Z_test)

        fit_obj3.fit(Z_train, t_train)
        preds3 = fit_obj3.predict_proba(Z_test)

        fit_obj4.fit(X_train2, y_train2)
        preds4 = fit_obj4.predict(X_test2)

        self.assertTrue(np.allclose(preds1[0, 0], 0.043789295499125226))

        self.assertTrue(np.allclose(preds1[0, 1], 0.9562107045008749))

        self.assertTrue(np.allclose(preds2[0, 0], 0.04650031359722393))

        self.assertTrue(np.allclose(preds3[0, 0], 0.04650031359722393))

        self.assertTrue(
            np.allclose(fit_obj.predict(X_test)[0], 1)
            & np.allclose(fit_obj2.predict(X_test)[0], 0)
        )

        self.assertTrue(np.allclose(preds4[0], 1.6404241349742967))

    def test_score(self):

        breast_cancer = load_breast_cancer()
        X = breast_cancer.data
        y = breast_cancer.target

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=123
        )

        clf = DecisionTreeClassifier(max_depth=1, random_state=14235)
        fit_obj = ns.RandomBagClassifier(
            clf,
            n_estimators=3,
            n_hidden_features=2,
            activation_name="relu",
            a=0.01,
            nodes_sim="sobol",
            bias=True,
            dropout=0,
            direct_link=False,
            n_clusters=2,
            type_clust="kmeans",
            type_scaling=("std", "std", "std"),
            col_sample=0.9,
            row_sample=0.9,
            n_jobs=None,
            seed=14253,
            verbose=0,
        )

        clf2 = DecisionTreeClassifier(max_depth=1, random_state=15243)
        fit_obj2 = ns.RandomBagClassifier(
            clf2,
            n_estimators=3,
            n_hidden_features=2,
            activation_name="relu",
            a=0.01,
            nodes_sim="sobol",
            bias=True,
            dropout=0,
            direct_link=False,
            n_clusters=2,
            type_clust="gmm",
            type_scaling=("std", "std", "std"),
            col_sample=0.9,
            row_sample=0.9,
            n_jobs=None,
            seed=19237,
            verbose=0,
        )

        fit_obj.fit(X_train, y_train)
        score1 = fit_obj.score(X_test, y_test)

        fit_obj2.fit(X_train, y_train)
        score2 = fit_obj2.score(X_test, y_test)

        self.assertTrue(
            np.allclose(score1, 0.9385964912280702)
            & np.allclose(score2, 0.9298245614035088)
        )


if __name__ == "__main__":
    ut.main()
