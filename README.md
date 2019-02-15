# nnetsauce 

This package does Machine Learning by using various -- advanced -- combinations of single layer neural networks. 


## Installation 

- __1st method__: from Github

```bash
git clone https://github.com/thierrymoudiki/nnetsauce.git
cd nnetsauce
python setup.py install
```

- __2nd method__: by using `pip`

```bash
TODO
```


## Dependencies 

- Numpy
- Scipy
- scikit-learn


## Package description

Every model in the `nnetsauce` is based on the component __g(XW + b)__, where:

- __X__ is a matrix containing the explanatory variables and (optional) clustering information about the individuals. The clustering methods available are _k-means_ and a _Gaussian Mixture Model_; they help in taking into account data's heterogeneity.
- __W__ derives the nodes in the hidden layer from __X__. It can be drawn from various random and quasirandom sequences.
- __b__ is an optional bias parameter.
- __g__ is an activation function such as the hyperbolic tangent (among others), which creates new, nonlinear explanatory variables.  

__Currently__, 4 models are implemented in the `nnetsauce`. If your response variable is __y__, then:

- `Base` adjusts a linear regression to __y__, as a function of __X__ (optional) and __g(XW + b)__; without regularization of the regression coefficients. 
- `BayesianRVFL` adds a ridge regularization parameter to the regression coefficients of `Base`, which prevents overfitting. Confidence intervals around the prediction can also be obtained.  
- `BayesianRVFL2` adds 2 regularization parameters to `Base`. As with `BayesianRVFL`, confidence intervals around the prediction can be obtained.
- `Custom` works with any object `fit_obj` possessing methods `fit_obj.fit()` and `fit_obj.predict()`. Notably, the model can be applied to any [`scikit-learn`](https://scikit-learn.org)'s regression or classification model. It adjusts `fit_obj` to __y__, as a function of __X__ (optional) and __g(XW + b)__.


## Short demo

Here, we present examples of use of `Base`, `BayesianRVFL`, `BayesianRVFL2`, and an example of `Custom` model using `scikit-learn`. We start by importing the packages and datasets necessary for the demo:

````python
import nnetsauce as ns
import numpy as np      
import matplotlib.pyplot as plt
from sklearn import datasets

diabetes = datasets.load_diabetes()
X = diabetes.data 
y = diabetes.target

breast_cancer = datasets.load_breast_cancer()
Z = breast_cancer.data
t = breast_cancer.target
````

Example with `Base` model (no regularization):

````python
# create object Base 

# X is not used directly, but only g(XW+b) ('direct_link')
# W is drawn from a deterministic Sobol sequence ('nodes_sim')
# b is equal to 0 ('bias')
# The activation function g is the hyperbolic tangent ('activation_name')
# The data in X is clustered: 2 clusters are obtained with k-means before fitting the model ('type_clust', 'n_clusters')

fit_obj = ns.Base(n_hidden_features=100, 
                  direct_link=False,
                  nodes_sim='sobol',
                  bias=False,
                  activation_name='tanh', 
                  type_clust='kmeans',
                  n_clusters=2) 

# fit training set 
fit_obj.fit(X[0:350,:], y[0:350])

# predict on test set 
x = np.linspace(351, 442, num = 442-351+1)
plt.scatter(x = x, y = y[350:442], color='black')
plt.plot(x, fit_obj.predict(X[350:442,:]), color='red')
plt.title('preds vs test set obs')
plt.xlabel('x')
plt.ylabel('preds')
plt.show()
````

Example with `BayesianRVFL` model (one regularization parameter):

````python
# create object BayesianRVFL  
# regularization is controlled by 's' and 'sigma'
# here, nodes_sim='halton' is used
fit_obj = ns.BayesianRVFL(n_hidden_features=100,
                          nodes_sim='halton', 
                          direct_link=True,
                          bias=False,
                          activation_name='relu',
                          n_clusters=3, s=0.01,sigma=0.1)

# fit training set 
fit_obj.fit(X[0:350,:], y[0:350])

# predict on test set 
x = np.linspace(351, 375, num = 375-351+1)
plt.scatter(x = x, y = y[350:375], color='black')
plt.plot(x, fit_obj.predict(X[350:375,:])[0], color='blue')
plt.title('preds vs test set obs')
plt.xlabel('x')
plt.ylabel('preds')
plt.show()
````

Example with `BayesianRVFL2` model (two regularization parameters):

````python
# create object BayesianRVFL2 
# regularization is controlled by 's1', 's2' and 'sigma'
fit_obj = ns.BayesianRVFL2(n_hidden_features=100, 
                  direct_link=True,
                  activation_name='tanh', 
                  n_clusters=3, 
                  s1=0.5, s2=0.1, sigma=0.1)

# fit training set 
fit_obj.fit(X[0:350,:], y[0:350])

# predict on test set 
x = np.linspace(351, 375, num = 375-351+1)
plt.scatter(x = x, y = y[350:375], color='black')
plt.plot(x, fit_obj.predict(X[350:375,:])[0], color='blue')
plt.title('preds vs test set obs')
plt.xlabel('x')
plt.ylabel('preds')
plt.show()
````

Example of `Custom` models, using `scikit-learn`:

````python
from sklearn import linear_model, gaussian_process

regr = linear_model.BayesianRidge()
regr2 = linear_model.ElasticNet()
regr3 = gaussian_process.GaussianProcessClassifier()

# create object Custom 
fit_obj = ns.Custom(obj=regr, n_hidden_features=100, 
                    direct_link=False, bias=True,
                    activation_name='tanh', n_clusters=2)

fit_obj2 = ns.Custom(obj=regr2, n_hidden_features=500, 
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

# predict classes 
print(fit_obj3.predict(Z[456:569,:]))
````

## Model validation

Currently, a method `score` is available for all models in the `nnetsauce`. It allows to measure the model's 
performance on a given testing set __(X, y)__. The `scoring` options are the same 
as `scikit-learn`'s. Using the previous code snippet, we have: 

````python
# RMSE of BayesianRidge on test set (X[350:360,:], y[350:360])
print(np.sqrt(fit_obj.score(X[350:360,:], y[350:360], 
        scoring="neg_mean_squared_error")))

# RMSE of ElasticNet on test set (X[350:360,:], y[350:360])
print(np.sqrt(fit_obj2.score(X[350:360,:], y[350:360], 
        scoring="neg_mean_squared_error")))

# Accuracy of Gaussian Process classifier on test set
print(fit_obj3.score(Z[456:569,:], t[456:569], scoring="accuracy"))
````

For `BayesianRVFL` and `BayesianRVFL2` we also have the Generalized Cross-Validation (GCV) 
error calculated right after the model is fitted:

````python
# Obtain GCV 
fit_obj = ns.BayesianRVFL2(n_hidden_features=100, 
                  direct_link=True,
                  activation_name='tanh', 
                  n_clusters=3, 
                  s1=0.5, s2=0.1, sigma=0.1)

fit_obj.fit(X[0:350,:], y[0:350])

print(fit_obj.GCV)
````

For `Custom`, in addition to the method `score`, we have a cross-validation method `cross_val_score` similar to `scikit-learn` 's `cross_val_score`: 

````python
regr = linear_model.BayesianRidge()
regr3 = gaussian_process.GaussianProcessClassifier()

# create objects Custom
fit_obj = ns.Custom(obj=regr, n_hidden_features=100, 
                    direct_link=True, bias=True,
                    activation_name='tanh', n_clusters=2)

fit_obj3 = ns.Custom(obj=regr3, n_hidden_features=100, 
                     direct_link=True, bias=True,
                     activation_name='relu', n_clusters=0)

# 5-fold cross-validation error (regression)
print(fit_obj.cross_val_score(X, y, cv = 5))

# 5-fold cross-validation error (classification)
print(fit_obj3.cross_val_score(Z, t, cv = 5))
````


## References

- Jones E, Oliphant E, Peterson P, et al. SciPy: Open Source Scientific Tools for Python, 2001-, http://www.scipy.org/ [Online; accessed 2019-01-04]
- Scikit-learn: Machine Learning in Python, Pedregosa et al., JMLR 12, pp. 2825-2830, 2011.


## License

[BSD 3-Clause](LICENSE.txt) © Thierry Moudiki, 2019. 
