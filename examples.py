import nnetsauce as ns
from sklearn import datasets, linear_model
import matplotlib.pyplot as plt  
import numpy as np 


## 0 - Data -----

diabetes = datasets.load_diabetes()

# define X and y
X = diabetes.data 
y = diabetes.target

 
## Example 1 - Base - n_hidden_features=5 -----

# create object Base 
fit_obj = ns.Base(n_hidden_features=5, 
                  direct_link=False,
                  activation_name='tanh', 
                  n_clusters=3)

fit_obj2 = ns.Base(n_hidden_features=5, 
                  direct_link=True,
                  activation_name='relu', 
                  n_clusters=3)

fit_obj3 = ns.Base(n_hidden_features=5, 
                   direct_link=True,
                   activation_name='tanh', 
                   n_clusters=3)    

# fit training set 
fit_obj.fit(X[0:350,:], y[0:350])
fit_obj2.fit(X[0:350,:], y[0:350])
fit_obj3.fit(X[0:350,:], y[0:350])

# predict on test set 
x = np.linspace(351, 442, num = 442-351+1)
plt.scatter(x = x, y = y[350:442], color='black')
plt.plot(x, fit_obj.predict(X[350:442,:]), color='red')
plt.plot(x, fit_obj2.predict(X[350:442,:]), color='blue')
plt.plot(x, fit_obj3.predict(X[350:442,:]), color='green')
plt.title('preds vs obs')
plt.xlabel('x')
plt.ylabel('preds')
plt.show()


## Example 2 - Base - n_hidden_features=5 -----

# create object Base 
fit_obj = ns.Base(n_hidden_features=100, 
                  direct_link=False,
                  bias=False,
                  activation_name='tanh', 
                  n_clusters=2)

fit_obj2 = ns.Base(n_hidden_features=100, 
                  direct_link=True,
                  bias=False,
                  activation_name='relu', 
                  n_clusters=2)

fit_obj3 = ns.Base(n_hidden_features=100, 
                   direct_link=True,
                   bias=False,
                   activation_name='tanh', 
                   n_clusters=2)    

# fit training set 
fit_obj.fit(X[0:350,:], y[0:350])
fit_obj2.fit(X[0:350,:], y[0:350])
fit_obj3.fit(X[0:350,:], y[0:350])

# predict on test set 
x = np.linspace(351, 442, num = 442-351+1)
plt.scatter(x = x, y = y[350:442], color='black')
plt.plot(x, fit_obj.predict(X[350:442,:]), color='red')
plt.plot(x, fit_obj2.predict(X[350:442,:]), color='blue')
plt.plot(x, fit_obj3.predict(X[350:442,:]), color='green')
plt.title('preds vs obs')
plt.xlabel('x')
plt.ylabel('preds')
plt.show()


## Example 3 - Custom - n_hidden_features=5 -----

regr = linear_model.BayesianRidge()
regr2 = linear_model.ElasticNet()

# create object Base 
fit_obj = ns.Custom(regr = regr, n_hidden_features=100, 
                    direct_link=False, bias=False,
                    activation_name='tanh', n_clusters=2)

fit_obj2 = ns.Custom(regr = regr2, n_hidden_features=500, 
                    direct_link=True, bias=False,
                    activation_name='tanh', n_clusters=0)

# fit training set 
fit_obj.fit(X[0:350,:], y[0:350])
fit_obj2.fit(X[0:350,:], y[0:350])

# predict on test set 
x = np.linspace(351, 442, num = 442-351+1)
plt.scatter(x = x, y = y[350:442], color='black')
plt.plot(x, fit_obj.predict(X[350:442,:]), color='red')
plt.plot(x, fit_obj2.predict(X[350:442,:]), color='blue')
plt.title('preds vs obs')
plt.xlabel('x')
plt.ylabel('preds')
plt.show()