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
obj2 = ns.GLMRegressor(n_hidden_features=3, 
                       lambda1=1e-2, alpha1=0.5,
                       lambda2=1e-2, alpha2=0.5,
                       optimizer=ns.optimizers.Optimizer(type_optim="sgd"))
start = time()

obj2.fit(X_train, y_train, learning_rate=0.1, batch_prop=0.25, verbose=0)
print(f"\n Elapsed: {time() - start}")
plt.plot(obj2.optimizer.results[2])
print(obj2.beta_)
print("RMSE: ")
print(np.sqrt(obj2.score(X_test, y_test))) # RMSE


print(f"\n Example 2 -----")
obj2.optimizer.type_optim = "scd"
start = time()
obj2.fit(X_train, y_train, learning_rate=0.01, batch_prop=0.8, verbose=0)
print(f"\n Elapsed: {time() - start}")
plt.plot(obj2.optimizer.results[2])
print(obj2.beta_)
print("RMSE: ")
print(np.sqrt(obj2.score(X_test, y_test))) # RMSE

print(f"\n Example 3 -----")
obj2.optimizer.type_optim = "sgd"
obj2.set_params(lambda1=1e-2, alpha1=0.1,
               lambda2=1e-1, alpha2=0.9)
start = time()
obj2.fit(X_train, y_train, batch_prop=0.25, verbose=0)
print(f"\n Elapsed: {time() - start}")
plt.plot(obj2.optimizer.results[2])
print(obj2.beta_)
print("RMSE: ")
print(np.sqrt(obj2.score(X_test, y_test))) # RMSE

print(f"\n Example 4 -----")
obj2.optimizer.type_optim = "scd"
start = time()
obj2.fit(X_train, y_train, learning_rate=0.01, batch_prop=0.8, verbose=0)
print(f"\n Elapsed: {time() - start}")
plt.plot(obj2.optimizer.results[2])
print(obj2.beta_)
print("RMSE: ")
print(np.sqrt(obj2.score(X_test, y_test))) # RMSE 

print(f"\n Example 5 -----")
obj2.optimizer.type_optim = "sgd"
obj2.set_params(lambda1=1, alpha1=0.5,
               lambda2=1e-2, alpha2=0.1)
start = time()
obj2.fit(X_train, y_train, learning_rate=0.1, batch_prop=0.5, verbose=0)
print(f"\n Elapsed: {time() - start}")
plt.plot(obj2.optimizer.results[2])
print(obj2.beta_)
print("RMSE: ")
print(np.sqrt(obj2.score(X_test, y_test))) # RMSE

print(f"\n Example 6 -----")
obj2.optimizer.type_optim = "scd"
start = time()
obj2.fit(X_train, y_train, learning_rate=0.1, batch_prop=0.5, verbose=0)
print(f"\n Elapsed: {time() - start}")
plt.plot(obj2.optimizer.results[2])
print(obj2.beta_)
print("RMSE: ")
print(np.sqrt(obj2.score(X_test, y_test))) # RMSE

