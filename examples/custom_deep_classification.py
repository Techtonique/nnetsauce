import os 
import nnetsauce as ns 
import sklearn.metrics as skm2
from sklearn.datasets import load_breast_cancer, load_iris, load_wine, load_digits
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split

print(f"\n ----- Running: {os.path.basename(__file__)}... ----- \n")

data = load_wine()
X = data.data
y= data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .2, random_state = 123)

obj = LogisticRegressionCV()

clf = ns.DeepClassifier(obj, n_layers=4, verbose=0, n_clusters=2)

clf.fit(X_train, y_train)

preds = clf.predict(X_test)

print(clf.score(X_test, y_test))
