import os 
import nnetsauce as ns 
from sklearn.datasets import load_breast_cancer, load_iris, load_wine, load_digits
from sklearn.model_selection import train_test_split
from time import time 

print(f"\n ----- Running: {os.path.basename(__file__)}... ----- \n")

load_models = [load_breast_cancer, load_iris, load_wine, load_digits]

# without preprocessing

for model in load_models: 

    print(f"\n Calling {model.__name__}")
    data = model()
    X = data.data
    y= data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .3, random_state = 13)

    clf = ns.LazyDeepClassifier(n_layers=3, verbose=0, 
                                ignore_warnings=True, 
                                n_hidden_features=3, 
                                estimators=["RandomForestClassifier", 
                                            "ExtraTreesClassifier", 
                                            "RandomForestRegressor"])

    start = time()
    models = clf.fit(X_train, X_test, y_train, y_test)
    print(f"\nElapsed: {time() - start} seconds\n")

    print(models)

for model in load_models: 

    print(f"\n Calling {model.__name__}")
    data = model()
    X = data.data
    y= data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .3, random_state = 13)

    clf = ns.LazyDeepClassifier(n_layers=3, verbose=0, 
                                ignore_warnings=True, 
                                n_hidden_features=3)

    start = time()
    models = clf.fit(X_train, X_test, y_train, y_test)
    print(f"\nElapsed: {time() - start} seconds\n")

    print(models)

# with preprocessing
    
for model in load_models: 

    print(f"\n Calling {model.__name__}")
    data = model()    
    X = data.data
    y= data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .3, random_state = 13)

    clf = ns.LazyDeepClassifier(n_layers=3, verbose=0, 
                                ignore_warnings=True,                                 
                                n_hidden_features=10, 
                                estimators="all",                                                        
                                preprocess = True)

    start = time()
    models = clf.fit(X_train, X_test, y_train, y_test)
    print(f"\nElapsed: {time() - start} seconds\n")

    print(models)
