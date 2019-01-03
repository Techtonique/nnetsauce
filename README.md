# nnetsauce 

This package does Machine Learning by using various (advanced) combinations of single layer neural networks. Every model in `nnetsauce` is based on the component $g(XW + b)$, where:

- $X$ is a matrix containing the explanatory variables and (optional) clustering information about the individuals. The clustering methods available are the _k-means_ and a _Gaussian Mixture Model_.
- $W$ constructs the nodes in the hidden layer from $X$
- $b$ is an optional bias parameter
- $g$ is an activation function such as the hyperbolic tangent (among others)  


## Installation 

- __1st method__: from Github

```
git clone https://github.com/thierrymoudiki/nnetsauce.git
cd nnetsauce
python setup.py install
```

- __2nd method__: by using `pip`

```
TODO
```


## Dependencies 

- Numpy
- Scipy
- scikit-learn


## Package description

__Currently__, $4$ models are implemented in the `nnetsauce`. If your response variable is $y$, then:

- `Base` adjusts a linear regression to $y$, as a function of $X$ (optional) and $g(XW + b)$; without regularization of the parameters. 
- `BayesianRVFL` adds a regularization parameter to the regression coefficients of `Base`, which prevents overfitting. Confidence intervals around the prediction can also be obtained.  
- `BayesianRVFL2` adds $2$ regularization parameters to `Base`. As with `BayesianRVFL`, confidence intervals around the prediction can be obtained.
- `Custom` works with any object `fit_obj` possessing the methods `fit_obj.fit()` and `fit_obj.predict()`. Notably, the model can be applied to any `scikit-learn`'s regression or classification model. It adjusts `fit_obj` to $y$, as a function of $X$ (optional) and $g(XW + b)$.


## Short demo

Here, we present examples of use of `Base`, `BayesianRVFL`, `BayesianRVFL2`, and an example of `Custom` model using `scikit-learn`. We start by importing the packages and dataset necessary for the demo:

````
import nnetsauce as ns
from sklearn import datasets

diabetes = datasets.load_diabetes()
X = diabetes.data 
y = diabetes.target
````

Example with `Base` model (no regularization):

````
# create object Base 
# type ?ns.Base for help on model parameters 
fit_obj = ns.Base(n_hidden_features=100, 
                  direct_link=False,
                  bias=False,
                  activation_name='tanh', 
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

````
# create object BayesianRVFL 
# type ?ns.BayesianRVFL for help on model parameters 
# regularization is controlled by 's' and 'sigma'
fit_obj = ns.BayesianRVFL(n_hidden_features=100, 
                  direct_link=True,
                  bias=False,
                  activation_name='relu', 
                  n_clusters=3, s=0.01, sigma=0.1)

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

````
# create object BayesianRVFL2 
# type ?ns.BayesianRVFL2 for help on model parameters 
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

Example with `Custom` model, using `scikit-learn`:

````
from sklearn import linear_model

regr = linear_model.BayesianRidge()
regr2 = linear_model.ElasticNet()

# create object Base 
fit_obj = ns.Custom(obj=regr, n_hidden_features=100, 
                    direct_link=False, bias=True,
                    activation_name='tanh', n_clusters=2)

fit_obj2 = ns.Custom(obj=regr2, n_hidden_features=500, 
                    direct_link=True, bias=False,
                    activation_name='relu', n_clusters=0)

# fit training set 
fit_obj.fit(X[0:350,:], y[0:350])
fit_obj2.fit(X[0:350,:], y[0:350])

# predict on test set 
x = np.linspace(351, 442, num = 442-351+1)
plt.scatter(x = x, y = y[350:442], color='black')
plt.plot(x, fit_obj.predict(X[350:442,:]), color='red')
plt.plot(x, fit_obj2.predict(X[350:442,:]), color='blue')
plt.title('preds vs test set obs')
plt.xlabel('x')
plt.ylabel('preds')
plt.show()
````

## Model validation

Currently, a method `score` is available for all the models in the `nnetsauce`. It allows to measure its 
performance on a given testing set $(X, y)$. The `scoring` options are the same 
as `scikit-learn`'s. Using the previous code snippet, we have: 

````
# RMSE of BayesianRidge on test set (X[350:360,:], y[350:360])
np.sqrt(fit_obj.score(X[350:360,:], y[350:360], 
        scoring="neg_mean_squared_error"))

# RMSE of ElasticNet on test set (X[350:360,:], y[350:360])
np.sqrt(fit_obj2.score(X[350:360,:], y[350:360], 
        scoring="neg_mean_squared_error"))
````

For `Base`, `BayesianRVFL` and `BayesianRVFL2` we also have the Generalized Cross-Validation (GCV) 
error calculated right after the model is fitted:

````
fit_obj = ns.BayesianRVFL2(n_hidden_features=100, 
                  direct_link=True,
                  activation_name='tanh', 
                  n_clusters=3, 
                  s1=0.5, s2=0.1, sigma=0.1)

fit_obj.fit(X[0:350,:], y[0:350])

print(fit_obj.GCV)

````

For `Custom`, in addition to the method `score`, we have `cross_val_score` such as  
`scikit-learn` 's `cross_val_score`: 

````
regr = linear_model.BayesianRidge()

# create object Custom
fit_obj = ns.Custom(obj = regr, n_hidden_features=100, 
                    direct_link=True, bias=True,
                    activation_name='tanh', n_clusters=2)

# 5-fold cross-validation
print(fit_obj.cross_val_score(X, y, cv=5))
````

## References

- Ref1
- Ref2
- Scikit-learn: Machine Learning in Python, Pedregosa et al., JMLR 12, pp. 2825-2830, 2011.

