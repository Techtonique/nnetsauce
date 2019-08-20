import numpy as np
from sklearn.linear_model import LogisticRegression
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
            X, y, test_size=0.2
        )

        wine = load_wine()
        Z = wine.data
        t = wine.target
        np.random.seed(123)
        Z_train, Z_test, t_train, t_test = train_test_split(
            Z, t, test_size=0.2
        )
        
        clf = LogisticRegression(solver='liblinear', multi_class = 'ovr')
        fit_obj = ns.AdaBoostClassifier(clf, 
                                n_hidden_features=np.int(11.22338867), 
                                direct_link=True,
                                n_estimators=5, learning_rate=0.01126343,
                                col_sample=0.72684326, row_sample=0.86429443,
                                dropout=0.63078613, n_clusters=0,
                                verbose=1, seed = 123, 
                                method="SAMME.R"
        )

        clf2 = LogisticRegression(solver='liblinear', multi_class = 'ovr')
        fit_obj2 = ns.AdaBoostClassifier(clf2, 
                                n_hidden_features=np.int(8.21154785e+01), 
                                direct_link=True,
                                n_estimators=5, learning_rate=2.96252441e-02,
                                col_sample=4.22766113e-01, row_sample=7.87268066e-01,
                                dropout=1.56909180e-01, n_clusters=0,
                                verbose=1, seed = 123, 
                                method="SAMME")  
        
        fit_obj.fit(X_train, y_train)
        preds1 = fit_obj.predict_proba(X_test)

        fit_obj2.fit(Z_train, t_train)
        preds2 = fit_obj2.predict_proba(Z_test)        

        self.assertFalse(
            np.allclose(preds1[0, 0], 0.0)
            & np.allclose(preds1[0, 1], 0.0)
            & np.allclose(preds2[0, 0], 0.0)
            & np.allclose(preds2[0, 1], 0.0)            
        )

        self.assertFalse(
            np.allclose(fit_obj.predict(X_test)[0], 1)
            & np.allclose(fit_obj2.predict(X_test)[0], 1)
        )

    def test_score(self):
        
        breast_cancer = load_breast_cancer()
        X = breast_cancer.data
        y = breast_cancer.target
        np.random.seed(123)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2
        )
        
        clf = LogisticRegression(solver='liblinear', multi_class = 'ovr')
        fit_obj = ns.AdaBoostClassifier(clf,
                                        n_hidden_features=np.int(11.22338867), 
                                        direct_link=True,
                                        n_estimators=5, learning_rate=0.01126343,
                                        col_sample=0.72684326, row_sample=0.86429443,
                                        dropout=0.63078613, n_clusters=0,
                                        verbose=1, seed = 123, 
                                        method="SAMME.R")

        clf2 = LogisticRegression(solver='liblinear', multi_class = 'ovr')
        fit_obj2 = ns.AdaBoostClassifier(clf2, 
                                n_hidden_features=np.int(8.21154785e+01), 
                                direct_link=True,
                                n_estimators=5, learning_rate=2.96252441e-02,
                                col_sample=4.22766113e-01, row_sample=7.87268066e-01,
                                dropout=1.56909180e-01, n_clusters=0,
                                verbose=1, seed = 123, 
                                method="SAMME")

        fit_obj.fit(X_train, y_train)
        score1 = fit_obj.score(X_test, y_test)

        fit_obj2.fit(X_train, y_train)
        score2 = fit_obj2.score(X_test, y_test)


        self.assertFalse(
            np.allclose(score1, 0.0)
            & np.allclose(score2, 0.0)
        )


if __name__ == "__main__":
    ut.main()
