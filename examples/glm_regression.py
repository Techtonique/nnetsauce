import numpy as np 
import nnetsauce as ns
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from time import time
import matplotlib.pyplot as plt


diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=2020)

print(f"\n Example 1 -----")
opt = ns.Optimizer(type_optim="sgd", 
                    learning_rate=0.1, 
                    batch_prop=0.25, 
                    verbose=0)
obj = ns.GLMRegressor(n_hidden_features=3, 
                       lambda1=1e-2, alpha1=0.5,
                       lambda2=1e-2, alpha2=0.5, 
                       optimizer=opt)

start = time()
obj.fit(X_train, y_train)
print(f"\n Elapsed: {time() - start}")
plt.plot(obj.optimizer.results[2])
print(obj.beta_)
print("RMSE: ")
print(np.sqrt(obj.score(X_test, y_test))) # RMSE


print(f"\n Example 2 -----")
opt2 = ns.Optimizer(type_optim="scd", 
                    learning_rate=0.01, 
                    batch_prop=0.8, 
                    verbose=1)
obj2 = ns.GLMRegressor(n_hidden_features=5, 
                       lambda1=1e-2, alpha1=0.5,
                       lambda2=1e-2, alpha2=0.5, 
                       optimizer=opt2)

start = time()
obj2.fit(X_train, y_train)
print(f"\n Elapsed: {time() - start}")
plt.plot(obj2.optimizer.results[2])
print(obj2.beta_)
print("RMSE: ")
print(np.sqrt(obj2.score(X_test, y_test))) # RMSE

print(f"\n Example 3 -----")

opt3 = ns.Optimizer(type_optim="scd",                     
                    batch_prop=0.25, 
                    verbose=1)
obj3 = ns.GLMRegressor(n_hidden_features=5, 
                       lambda1=1e-2, alpha1=0.1,
                       lambda2=1e-1, alpha2=0.9, 
                       optimizer=opt3)
start = time()
obj3.fit(X_train, y_train)
print(f"\n Elapsed: {time() - start}")
plt.plot(obj3.optimizer.results[2])
print(obj3.beta_)
print("RMSE: ")
print(np.sqrt(obj2.score(X_test, y_test))) # RMSE

print(f"\n Example 4 -----")

opt4 = ns.Optimizer(type_optim="scd", 
                    learning_rate=0.01,
                    batch_prop=0.8, 
                    verbose=0)
obj4 = ns.GLMRegressor(optimizer=opt4)

start = time()
obj4.fit(X_train, y_train)
print(f"\n Elapsed: {time() - start}")
plt.plot(obj4.optimizer.results[2])
print(obj4.beta_)
print("RMSE: ")
print(np.sqrt(obj4.score(X_test, y_test))) # RMSE 

print(f"\n Example 5 -----")

opt5 = ns.Optimizer(type_optim="scd", 
                    learning_rate=0.1,
                    batch_prop=0.5, 
                    verbose=0)
obj5 = ns.GLMRegressor(optimizer=opt5, 
                       lambda1=1, 
                       alpha1=0.5, 
                       lambda2=1e-2, 
                       alpha2=0.1)

start = time()
obj5.fit(X_train, y_train)
print(f"\n Elapsed: {time() - start}")
plt.plot(obj5.optimizer.results[2])
print(obj5.beta_)
print("RMSE: ")
print(np.sqrt(obj5.score(X_test, y_test))) # RMSE
