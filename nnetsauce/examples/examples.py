from sklearn import datasets, linear_model, gaussian_process, metrics
import matplotlib.pyplot as plt  
import numpy as np 
#import nnetsauce as ns

## 0 - Data -----

# define X and y
X, y = datasets.make_regression(n_samples=10, 
                                n_features=4)

# define X and y
diabetes = datasets.load_diabetes()
X = diabetes.data 
y = diabetes.target

breast_cancer = datasets.load_breast_cancer()
Z = breast_cancer.data
t = breast_cancer.target

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

plt.scatter(y[350:442], fit_obj.predict(X[350:442,:]), color='red')
plt.scatter(y[350:442], fit_obj2.predict(X[350:442,:]), color='blue')
plt.scatter(y[350:442], fit_obj3.predict(X[350:442,:]), color='green')
plt.scatter(x = np.sort(y[350:442]), y = np.sort(y[350:442]), 
         color='black')
plt.title('actual vs preds')
plt.xlabel('actual')
plt.ylabel('preds')
plt.show()

print("\n")
print("----- Example 1: Base ------")
print("\n")

print("fit_obj RMSE")
print( np.sqrt(((fit_obj.predict(X[350:442,:]) - y[350:442])**2).mean()))
print("\n")

print("fit_obj2 RMSE")
print( np.sqrt(((fit_obj2.predict(X[350:442,:]) - y[350:442])**2).mean()))
print("\n")

print("fit_obj3 RMSE")
print( np.sqrt(((fit_obj3.predict(X[350:442,:]) - y[350:442])**2).mean()))
print("\n")

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
plt.title('preds vs test set obs')
plt.xlabel('x')
plt.ylabel('preds')
plt.show()

print("\n")
print("----- Example 2: Base ------")
print("\n")


print("fit_obj RMSE")
print( np.sqrt(((fit_obj.predict(X[350:442,:]) - y[350:442])**2).mean()))
print("\n")

print("fit_obj2 RMSE")
print( np.sqrt(((fit_obj2.predict(X[350:442,:]) - y[350:442])**2).mean()))
print("\n")

print("fit_obj3 RMSE")
print( np.sqrt(((fit_obj3.predict(X[350:442,:]) - y[350:442])**2).mean()))
print("\n")

## Example 3 - Custom - n_hidden_features=100, 500 -----

regr = linear_model.BayesianRidge()
regr2 = linear_model.ElasticNet()
regr3 = gaussian_process.GaussianProcessClassifier()

# create object Base 
fit_obj = ns.Custom(obj = regr, n_hidden_features=100, 
                    direct_link=False, bias=False,
                    activation_name='tanh', n_clusters=2)

fit_obj2 = ns.Custom(obj = regr2, n_hidden_features=500, 
                    direct_link=True, bias=False,
                    activation_name='relu', n_clusters=0)

fit_obj3 = ns.Custom(obj = regr3, n_hidden_features=100, 
                    direct_link=True, bias=True,
                    activation_name='relu', n_clusters=0)

# fit training set 
fit_obj.fit(X[0:350,:], y[0:350])
fit_obj2.fit(X[0:350,:], y[0:350])
fit_obj3.fit(Z[0:455,:], t[0:455])

# predict on test set 
x = np.linspace(351, 442, num = 442-351+1)
plt.scatter(x = x, y = y[350:442], color='black')
plt.plot(x, fit_obj.predict(X[350:442,:]), color='red')
plt.plot(x, fit_obj2.predict(X[350:442,:]), color='blue')
plt.title('preds vs test set obs')
plt.xlabel('x')
plt.ylabel('preds')
plt.show()


fit_obj3.predict(Z[456:569,:])

print("\n")
print("----- Example 3: Custom ------")
print("\n")


print("fit_obj RMSE")
print( np.sqrt(((fit_obj.predict(X[350:442,:]) - y[350:442])**2).mean()))
print("\n")

print("fit_obj2 RMSE")
print( np.sqrt(((fit_obj2.predict(X[350:442,:]) - y[350:442])**2).mean()))
print("\n")


## Example 4 - BayesianRVFL2 - n_hidden_features=5 -----

# create object BayesianRVFL2 
fit_obj = ns.BayesianRVFL2(n_hidden_features=5, 
                  direct_link=False,
                  activation_name='tanh', 
                  n_clusters=3)

fit_obj2 = ns.BayesianRVFL2(n_hidden_features=5, 
                  direct_link=True,
                  activation_name='relu', 
                  n_clusters=3)

fit_obj3 = ns.BayesianRVFL2(n_hidden_features=5, 
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
plt.plot(x, fit_obj.predict(X[350:442,:])[0], color='red')
plt.plot(x, fit_obj2.predict(X[350:442,:])[0], color='blue')
plt.plot(x, fit_obj3.predict(X[350:442,:])[0], color='green')
plt.title('preds vs obs')
plt.xlabel('x')
plt.ylabel('preds')
plt.show()

print("\n")
print("----- Example 4: BayesianRVFL ------")
print("\n")


print("fit_obj RMSE")
print( np.sqrt(((fit_obj.predict(X[350:442,:])[0] - y[350:442])**2).mean()))
print("\n")

print("fit_obj2 RMSE")
print( np.sqrt(((fit_obj2.predict(X[350:442,:])[0] - y[350:442])**2).mean()))
print("\n")

print("fit_obj3 RMSE")
print( np.sqrt(((fit_obj3.predict(X[350:442,:])[0] - y[350:442])**2).mean()))
print("\n")

## Example 5 - BayesianRVFL - n_hidden_features=5 -----

# create object BayesianRVFL 
fit_obj = ns.BayesianRVFL(n_hidden_features=5, 
                  direct_link=False,
                  bias=False,
                  activation_name='tanh', 
                  n_clusters=2)

fit_obj2 = ns.BayesianRVFL(n_hidden_features=100, 
                  direct_link=True,
                  bias=False,
                  activation_name='relu', 
                  n_clusters=3)

fit_obj3 = ns.BayesianRVFL(n_hidden_features=100, 
                   direct_link=True,
                   bias=False,
                   activation_name='tanh', 
                   n_clusters=5)    

# fit training set 
fit_obj.fit(X[0:350,:], y[0:350])
fit_obj2.fit(X[0:350,:], y[0:350])
fit_obj3.fit(X[0:350,:], y[0:350])

# predict on test set 
x = np.linspace(351, 375, num = 375-351+1)
plt.scatter(x = x, y = y[350:375], color='black')
plt.plot(x, fit_obj.predict(X[350:375,:])[0], color='red')
plt.plot(x, fit_obj2.predict(X[350:375,:])[0], color='blue')
plt.plot(x, fit_obj3.predict(X[350:375,:])[0], color='green')
plt.title('preds vs test set obs')
plt.xlabel('x')
plt.ylabel('preds')
plt.show()

print("\n")
print("----- Example 5: BayesianRVFL ------")
print("\n")


print("fit_obj RMSE")
print( np.sqrt(((fit_obj.predict(X[350:442,:])[0] - y[350:442])**2).mean()))
print("\n")

print("fit_obj2 RMSE")
print( np.sqrt(((fit_obj2.predict(X[350:442,:])[0] - y[350:442])**2).mean()))
print("\n")

print("fit_obj3 RMSE")
print( np.sqrt(((fit_obj3.predict(X[350:442,:])[0] - y[350:442])**2).mean()))
print("\n")


# predict on test set 
x = np.linspace(351, 375, num = 375-351+1)
plt.scatter(x = x, y = y[350:375], color='black')
plt.plot(x, fit_obj.predict(X[350:375,:])[0], color='red')
plt.plot(x, fit_obj2.predict(X[350:375,:])[0], color='blue')
plt.plot(x, fit_obj3.predict(X[350:375,:])[0], color='green')
plt.title('preds vs test set obs')
plt.xlabel('x')
plt.ylabel('preds')
plt.show()


## Example 6 - BayesianRVFL2 - n_hidden_features=5 -----

# create object BayesianRVFL2 
fit_obj = ns.BayesianRVFL2(n_hidden_features=5, 
                  direct_link=False,
                  activation_name='tanh', 
                  n_clusters=3)

fit_obj2 = ns.BayesianRVFL2(n_hidden_features=5, 
                  direct_link=True,
                  activation_name='relu', 
                  n_clusters=3)

fit_obj3 = ns.BayesianRVFL2(n_hidden_features=5, 
                   direct_link=True,
                   activation_name='tanh', 
                   n_clusters=3)    

fit_obj4 = ns.BayesianRVFL2(n_hidden_features=5, 
                   direct_link=False,
                   bias=False,
                   activation_name='tanh', 
                   n_clusters=3)    

# fit training set 
fit_obj.fit(X[0:350,:], y[0:350])
fit_obj2.fit(X[0:350,:], y[0:350])
fit_obj3.fit(X[0:350,:], y[0:350])
fit_obj4.fit(X[0:350,:], y[0:350])

# predict on test set 
x = np.linspace(351, 442, num = 442-351+1)
plt.scatter(x = x, y = y[350:442], color='black')
plt.plot(x, fit_obj.predict(X[350:442,:])[0], color='red')
plt.plot(x, fit_obj2.predict(X[350:442,:])[0], color='blue')
plt.plot(x, fit_obj3.predict(X[350:442,:])[0], color='green')
plt.plot(x, fit_obj4.predict(X[350:442,:])[0], color='yellow')
plt.title('preds vs obs')
plt.xlabel('x')
plt.ylabel('preds')
plt.show()

print("\n")
print("----- Example 6: BayesianRVFL2 ------")
print("\n")


print("fit_obj RMSE")
print( np.sqrt(((fit_obj.predict(X[350:442,:])[0] - y[350:442])**2).mean()))
print("\n")

print("fit_obj2 RMSE")
print( np.sqrt(((fit_obj2.predict(X[350:442,:])[0] - y[350:442])**2).mean()))
print("\n")

print("fit_obj3 RMSE")
print( np.sqrt(((fit_obj3.predict(X[350:442,:])[0] - y[350:442])**2).mean()))
print("\n")

print("fit_obj4 RMSE")
print( np.sqrt(((fit_obj4.predict(X[350:442,:])[0] - y[350:442])**2).mean()))
print("\n")

## Example 5 - BayesianRVFL2 - n_hidden_features=5 -----

# create object BayesianRVFL2 
fit_obj = ns.BayesianRVFL2(n_hidden_features=5, 
                  direct_link=False,
                  bias=False,
                  activation_name='tanh', 
                  n_clusters=2)

fit_obj2 = ns.BayesianRVFL2(n_hidden_features=10, 
                  direct_link=True,
                  bias=False,
                  activation_name='relu', 
                  n_clusters=2)

fit_obj3 = ns.BayesianRVFL2(n_hidden_features=100, 
                   direct_link=True,
                   bias=False,
                   activation_name='tanh', 
                   n_clusters=2)    

fit_obj4 = ns.BayesianRVFL2(n_hidden_features=50, 
                  direct_link=False,
                  bias=False,
                  activation_name='tanh', 
                  n_clusters=4)

# fit training set 
fit_obj.fit(X[0:350,:], y[0:350])
fit_obj2.fit(X[0:350,:], y[0:350])
fit_obj3.fit(X[0:350,:], y[0:350])
fit_obj4.fit(X[0:350,:], y[0:350])

# predict on test set 
x = np.linspace(351, 375, num = 375-351+1)
plt.scatter(x = x, y = y[350:375], color='black')
plt.plot(x, fit_obj.predict(X[350:375,:])[0], color='red')
plt.plot(x, fit_obj2.predict(X[350:375,:])[0], color='blue')
plt.plot(x, fit_obj3.predict(X[350:375,:])[0], color='green')
plt.plot(x, fit_obj4.predict(X[350:375,:])[0], color='green')
plt.title('preds vs test set obs')
plt.xlabel('x')
plt.ylabel('preds')
plt.show()

print("\n")
print("----- Example 7: BayesianRVFL2 ------")
print("\n")


print("fit_obj RMSE")
print( np.sqrt(((fit_obj.predict(X[350:442,:])[0] - y[350:442])**2).mean()))
print("\n")

print("fit_obj2 RMSE")
print( np.sqrt(((fit_obj2.predict(X[350:442,:])[0] - y[350:442])**2).mean()))
print("\n")

print("fit_obj3 RMSE")
print( np.sqrt(((fit_obj3.predict(X[350:442,:])[0] - y[350:442])**2).mean()))
print("\n")


# predict on test set 
x = np.linspace(351, 375, num = 375-351+1)
plt.scatter(x = x, y = y[350:375], color='black')
plt.plot(x, fit_obj.predict(X[350:375,:])[0], color='red')
plt.plot(x, fit_obj2.predict(X[350:375,:])[0], color='blue')
plt.plot(x, fit_obj3.predict(X[350:375,:])[0], color='green')
plt.title('preds vs test set obs')
plt.xlabel('x')
plt.ylabel('preds')
plt.show()


# stacking layers - TODO - create function (?)
# stacking layers - TODO - create function (?)
# stacking layers - TODO - create function (?)
# stacking layers - TODO - create function (?)
# stacking layers - TODO - create function (?)


# layer 1 (base layer) ----
layer1_regr = linear_model.BayesianRidge()
layer1_regr.fit(X[0:100,:], y[0:100])
# RMSE
np.sqrt(metrics.mean_squared_error(y[100:125], layer1_regr.predict(X[100:125,:])))


# layer 2 ----
layer2_regr = ns.Custom(obj = layer1_regr, n_hidden_features=3, 
                        direct_link=True, bias=True, 
                        nodes_sim='sobol', activation_name='tanh', 
                        n_clusters=2)
layer2_regr.fit(X[0:100,:], y[0:100])

# RMSE
np.sqrt(layer2_regr.score(X[100:125,:], y[100:125]))

# layer 3 ----
layer3_regr = ns.Custom(obj = layer2_regr, n_hidden_features=5, 
                        direct_link=True, bias=True, 
                        nodes_sim='hammersley', activation_name='sigmoid', 
                        n_clusters=2)
layer3_regr.fit(X[0:100,:], y[0:100])

# RMSE
np.sqrt(layer3_regr.score(X[100:125,:], y[100:125]))

## Example 6 - MTS -----

from sklearn import datasets, linear_model, gaussian_process
import matplotlib.pyplot as plt  
import numpy as np 

X = np.random.rand(10, 4)
regr4 = gaussian_process.GaussianProcessRegressor()
obj_MTS = ns.MTS(regr4, lags = 1, n_hidden_features=5, 
                 bias = False)
obj_MTS.fit(X)
print(obj_MTS.predict())
print(obj_MTS.predict(return_std = True))

regr5 = linear_model.BayesianRidge()
obj_MTS2 = ns.MTS(regr5, lags = 1, n_hidden_features=7, 
                 bias = True)
obj_MTS2.fit(X)
print(obj_MTS2.predict())
print(obj_MTS2.predict(return_std = True))


# change: return_std = True must be in method predict
# change: return_std = True must be in method predict
# change: return_std = True must be in method predict
# change: return_std = True must be in method predict
# change: return_std = True must be in method predict
regr6 = ns.BayesianRVFL()
obj_MTS3 = ns.MTS(regr6, lags = 1, n_hidden_features=3, 
                 bias = True)
obj_MTS3.fit(X)
print(obj_MTS3.predict())
print(obj_MTS3.predict(return_std = True))

regr7 = ns.BayesianRVFL2()
obj_MTS3 = ns.MTS(regr7, lags = 1)
obj_MTS3.fit(X)
print(obj_MTS3.predict())
print(obj_MTS3.predict(return_std = True))