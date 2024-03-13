import os 
import numpy as np
from sklearn import datasets, linear_model, gaussian_process
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, ExtraTreesRegressor, ExtraTreesClassifier
import unittest as ut
import nnetsauce as ns
import pandas as pd 


print(f"\n ----- Running: {os.path.basename(__file__)}... ----- \n")

class TestLazyCustom(ut.TestCase):
    def test_custom(self):
        
        # regr
        X, y = datasets.make_regression(n_samples=15, 
                                        n_features=3, 
                                        random_state=123)

        # clf 
        wine = datasets.load_wine()
        Z = wine.data
        t = wine.target

        # mts 
        url = "https://raw.githubusercontent.com/thierrymoudiki/mts-data/master/heater-ice-cream/ice_cream_vs_heater.csv"
        df = pd.read_csv(url)
        df.set_index('Month', inplace=True) 
        df.index.rename('date')
        idx_train = int(df.shape[0]*0.8)
        idx_end = df.shape[0]
        df_train = df.iloc[0:idx_train,]
        df_test = df.iloc[idx_train:idx_end,]

        regr_mts3 = ns.LazyMTS(verbose=1, ignore_warnings=False, custom_metric=None,
                      lags = 4, n_hidden_features=7, n_clusters=2,
                      show_progress=True, preprocess=False, 
                      estimators=["ExtraTreesRegressor", "RandomForestRegressor"])
        models, predictions = regr_mts3.fit(df_train, df_test)
        model_dictionary = regr_mts3.provide_models(df_train, df_test)
        print(f"models: {models}")
        print(model_dictionary["MTS(ExtraTreesRegressor)"])


        regr_mts = ns.LazyMTS(verbose=0, ignore_warnings=True, custom_metric=None,
                            lags = 4, n_hidden_features=7, n_clusters=2,
                            estimators=["RandomForestRegressor",
                                            "ExtraTreesRegressor"],                                
                            show_progress=False, preprocess=False)
        models, predictions = regr_mts.fit(df_train, df_test)
        model_dictionary = regr_mts.provide_models(df_train, df_test)        
        print(model_dictionary["MTS(ExtraTreesRegressor)"])

        self.assertTrue(np.allclose(0, 1))

        regr_mts3 = ns.LazyDeepMTS(verbose=0, ignore_warnings=True, custom_metric=None,
                            lags = 4, n_hidden_features=7, n_clusters=2,
                            estimators=["RandomForestRegressor",
                                            "ExtraTreesRegressor"],                                
                            show_progress=False, preprocess=True)
        models, predictions = regr_mts2.fit(df_train, df_test)
        model_dictionary = regr_mts.provide_models(df_train, df_test)        
        print(model_dictionary["MTS(ExtraTreesRegressor)"])

        self.assertTrue(np.allclose(0, 1))

        regr_mts4 = ns.LazyDeepMTS(verbose=0, ignore_warnings=True, custom_metric=None,
                            lags = 4, n_hidden_features=7, n_clusters=2,
                            estimators=["RandomForestRegressor",
                                            "ExtraTreesRegressor"],                                
                            show_progress=False, preprocess=False)
        models, predictions = regr_mts.fit(df_train, df_test)
        model_dictionary = regr_mts.provide_models(df_train, df_test)        
        print(model_dictionary["MTS(ExtraTreesRegressor)"])

        self.assertTrue(np.allclose(0, 1))

        regr_mts2 = ns.LazyMTS(verbose=0, ignore_warnings=True, custom_metric=None,
                            lags = 4, n_hidden_features=7, n_clusters=2,
                            estimators=["RandomForestRegressor",
                                            "ExtraTreesRegressor"],                                
                            show_progress=False, preprocess=True)
        models, predictions = regr_mts2.fit(df_train, df_test)
        model_dictionary = regr_mts.provide_models(df_train, df_test)
        print(f"models: {models}")
        print(model_dictionary["MTS(ExtraTreesRegressor)"])

        self.assertTrue(np.allclose(0, 1))

        regr = ns.LazyRegressor(verbose=0, ignore_warnings=True, 
                                custom_metric=None, preprocess=False, 
                                estimators=["RandomForestRegressor",
                                            "ExtraTreesRegressor"],                                
                                        n_jobs=-1)
        
        regr2 = ns.LazyDeepRegressor(verbose=0, ignore_warnings=True, 
                                custom_metric=None, preprocess=False, 
                                estimators=["RandomForestRegressor",
                                            "ExtraTreesRegressor"],                                
                                        n_jobs=-1)
        
        clf = ns.LazyClassifier(verbose=0, ignore_warnings=True, 
                                custom_metric=None, preprocess=False, 
                                estimators=["RandomForestClassifier",
                                            "ExtraTreesClassifier"],                                
                                        n_jobs=-1)                
        
        clf2 = ns.LazyDeepClassifier(verbose=0, ignore_warnings=True, 
                                custom_metric=None, preprocess=False, 
                                estimators=["RandomForestClassifier",
                                            "ExtraTreesClassifier"],                                
                                        n_jobs=-1)                

        index_train = range(10)
        index_test = range(10, 15)
        X_train = X[index_train, :]
        y_train = y[index_train]
        X_test = X[index_test, :]
        y_test = y[index_test]
        Z_train = Z[index_train, :]
        t_train = t[index_train]
        Z_test = Z[index_test, :]
        t_test = t[index_test]

        models, predictions = regr.fit(X_train, X_test, y_train, y_test)
        print(models.iloc[0, 1])                
        self.assertTrue(np.allclose(models.iloc[0, 1], 0.506205210012072))

        models, predictions = regr2.fit(X_train, X_test, y_train, y_test)
        print(models.iloc[0, 1])                
        self.assertTrue(np.allclose(models.iloc[0, 1], 0.5148730790761626))

        models, predictions = clf.fit(Z_train, Z_test, t_train, t_test)
        print(models.iloc[0, 1])                
        self.assertTrue(np.allclose(models.iloc[0, 1], 1.00))

        models, predictions = clf2.fit(Z_train, Z_test, t_train, t_test)
        print(models.iloc[0, 1])                
        self.assertTrue(np.allclose(models.iloc[0, 1], 1.00))
        

if __name__ == "__main__":
    ut.main()
