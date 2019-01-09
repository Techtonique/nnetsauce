from sklearn import datasets, linear_model, gaussian_process
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