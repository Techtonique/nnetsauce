import nnetsauce as ns
import numpy as np
from sklearn.datasets import load_breast_cancer, load_wine, load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from time import time


# dataset no. 1 ----------

# logistic reg
breast_cancer = load_breast_cancer()
Z = breast_cancer.data
t = breast_cancer.target
np.random.seed(123)
X_train, X_test, y_train, y_test = train_test_split(Z, t, test_size=0.2)

# # SAMME
# clf = LogisticRegression(solver='liblinear', multi_class = 'ovr', 
#                          random_state=123)
# fit_obj = ns.AdaBoostClassifier(clf, 
#                                 n_hidden_features=np.int(56.13806152), 
#                                 direct_link=True,
#                                 n_estimators=1000, learning_rate=0.09393372,
#                                 col_sample=0.52887573, row_sample=0.87781372,
#                                 dropout=0.10216064, n_clusters=2,
#                                 type_clust="gmm",
#                                 verbose=1, seed = 123, 
#                                 method="SAMME") 

# start = time() 
# fit_obj.fit(X_train, y_train) 
# print(time() - start)

# print(fit_obj.score(X_test, y_test))
# preds = fit_obj.predict(X_test)                        

# print(fit_obj.score(X_test, y_test, scoring="roc_auc"))
# print(metrics.classification_report(preds, y_test))

# SAMME.R
# clf = LogisticRegression(solver='liblinear', multi_class = 'ovr', 
#                          random_state=123)
# fit_obj = ns.AdaBoostClassifier(clf, 
#                                 n_hidden_features=np.int(11.22338867), 
#                                 direct_link=True,
#                                 n_estimators=250, learning_rate=0.01126343,
#                                 col_sample=0.72684326, row_sample=0.86429443,
#                                 dropout=0.63078613, n_clusters=2,
#                                 type_clust="gmm",
#                                 verbose=1, seed = 123, 
#                                 method="SAMME.R")  
# start = time() 
# fit_obj.fit(X_train, y_train) 
# print(f"Elapsed {time() - start}") 

# start = time() 
# print(fit_obj.score(X_test, y_test))
# print(f"Elapsed {time() - start}") 

# preds = fit_obj.predict(X_test)                        

# print(fit_obj.score(X_test, y_test, scoring="roc_auc"))
# print(metrics.classification_report(preds, y_test))


# dataset no. 2 ----------

wine = load_wine()
Z = wine.data
t = wine.target
np.random.seed(123)
Z_train, Z_test, y_train, y_test = train_test_split(Z, t, test_size=0.2)


# SAMME
clf = LogisticRegression(solver='liblinear', multi_class = 'ovr', 
                         random_state=123)
fit_obj = ns.AdaBoostClassifier(clf, 
                                n_hidden_features=np.int(8.21154785e+01), 
                                direct_link=True,
                                n_estimators=1000, learning_rate=2.96252441e-02,
                                col_sample=4.22766113e-01, row_sample=7.87268066e-01,
                                dropout=1.56909180e-01, n_clusters=3,
                                type_clust="gmm",
                                verbose=1, seed = 123, 
                                method="SAMME")  
start = time() 
fit_obj.fit(Z_train, y_train) 
print(f"Elapsed {time() - start}") 
start = time() 
print(fit_obj.score(Z_test, y_test))
print(f"Elapsed {time() - start}")  
preds = fit_obj.predict(Z_test)     
print(metrics.classification_report(preds, y_test))     


# dataset no. 3 ----------

iris = load_iris()
Z = iris.data
t = iris.target
np.random.seed(123)
Z_train, Z_test, y_train, y_test = train_test_split(Z, t, test_size=0.2)


# SAMME.R
clf = LogisticRegression(solver='liblinear', multi_class = 'ovr', 
                         random_state=123)
fit_obj = ns.AdaBoostClassifier(clf, 
                                n_hidden_features=np.int(19.66918945), 
                                direct_link=True,
                                n_estimators=250, learning_rate=0.28534302,
                                col_sample=0.45474854, row_sample=0.87833252,
                                dropout=0.15603027, n_clusters=0,
                                verbose=1, seed = 123, 
                                method="SAMME.R")  
start = time() 
fit_obj.fit(Z_train, y_train)
print(f"Elapsed {time() - start}")   
start = time() 
print(fit_obj.score(Z_test, y_test))
print(f"Elapsed {time() - start}")    
preds = fit_obj.predict(Z_test)     
print(metrics.classification_report(preds, y_test))     
                    


