import os 
import nnetsauce as ns 
from sklearn.datasets import load_breast_cancer, load_iris, load_wine, load_digits
from sklearn.model_selection import train_test_split

print(f"\n ----- Running: {os.path.basename(__file__)}... ----- \n")

data = load_breast_cancer()
X = data.data
y= data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .2, random_state = 123)

clf = ns.LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None, 
                        preprocess=True)
models, predictions = clf.fit(X_train, X_test, y_train, y_test)
model_dictionary = clf.provide_models(X_train, X_test, y_train, y_test)
print(models)
#print(model_dictionary["LogisticRegression"])

clf2 = ns.LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None, 
                        preprocess=False)
models, predictions = clf2.fit(X_train, X_test, y_train, y_test)
model_dictionary = clf2.provide_models(X_train, X_test, y_train, y_test)
print(models)