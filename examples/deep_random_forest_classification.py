
import os
import nnetsauce as ns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
from time import time

print(f"\n ----- Running: {os.path.basename(__file__)}... ----- \n")

# See also examples/custom_deep_*.py

digits = load_digits()
X = digits.data
y = digits.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=123)


# layer 1 (base layer) ----
print(" \n layer 1 ----- \n")
layer1_regr = RandomForestClassifier(n_estimators=10, random_state=123)

start = time() 

layer1_regr.fit(X_train, y_train)

# Accuracy in layer 1
print(layer1_regr.score(X_test, y_test))


# layer 2 using layer 1 ----
print(" \n layer 2 ----- \n")
layer2_regr = ns.CustomClassifier(obj = layer1_regr, n_hidden_features=5, 
                        direct_link=True, bias=True, 
                        nodes_sim='uniform', activation_name='relu', 
                        n_clusters=2, seed=123)
layer2_regr.fit(X_train, y_train)

# Accuracy in layer 2
print(layer2_regr.score(X_test, y_test))

print(f"Elapsed {time() - start}")  