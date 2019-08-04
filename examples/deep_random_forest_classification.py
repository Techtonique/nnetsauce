
import nnetsauce as ns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits


digits = load_digits()
X = digits.data
y = digits.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=123)


# layer 1 (base layer) ----
layer1_regr = RandomForestClassifier()
layer1_regr.fit(X_train, y_train)

# Accuracy in layer 1
print(layer1_regr.score(X_test, y_test))


# layer 2 using layer 1 ----
layer2_regr = ns.CustomClassifier(obj = layer1_regr, n_hidden_features=3, 
                        direct_link=True, bias=True, 
                        nodes_sim='sobol', activation_name='tanh', 
                        n_clusters=2)
layer2_regr.fit(X_train, y_train)

# Accuracy in layer 2
print(layer2_regr.score(X_test, y_test))


# layer 3 using layer 2 ----
layer3_regr = ns.CustomClassifier(obj = layer2_regr, n_hidden_features=5, 
                        direct_link=True, bias=True, dropout=0.9,
                        nodes_sim='sobol', activation_name='tanh', 
                        n_clusters=0)
layer3_regr.fit(X_train, y_train)

# Accuracy in layer 3
print(layer3_regr.score(X_test, y_test))
