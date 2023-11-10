import os 
import nnetsauce as ns 
from sklearn.datasets import load_breast_cancer, load_iris, load_wine, load_digits
from sklearn.model_selection import train_test_split

print(f"\n ----- Running: {os.path.basename(__file__)}... ----- \n")

data = load_digits()
X = data.data
y= data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .2, random_state = 123)

clf = ns.LazyDeepClassifier(n_layers=3, verbose=0, ignore_warnings=True)
models, predictions = clf.fit(X_train, X_test, y_train, y_test)
model_dictionary = clf.provide_models(X_train, X_test, y_train, y_test)

print(models)