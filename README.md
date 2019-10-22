![nnetsauce logo](the-nnetsauce.png)

<hr>

This package does Statistical/Machine Learning by using various -- advanced -- combinations of single layer 'neural' networks. 


## Installation 

- __1st method__: by using `pip` at the command line for the stable version

```bash
pip install nnetsauce
```


- __2nd method__: from Github, for the development version

```bash
pip install git+https://github.com/thierrymoudiki/nnetsauce.git
```



## Package description

Every model in the `nnetsauce` is based on the component __g(XW + b)__, where:

- __X__ is a matrix containing the explanatory variables and (optional) clustering information. The clustering methods available are _k-means_ and a _Gaussian Mixture Model_; they help in taking into account input data's heterogeneity.
- __W__ creates new, additional explanatory variables from __X__. It can be drawn from various random and quasirandom sequences.
- __b__ is an optional bias parameter.
- __g__ is an _activation function_ such as the hyperbolic tangent or the sigmoid function (among others), that renders the combination of explanatory variables (through __W__) nonlinear.  

__Currently__, 5 models are implemented in the `nnetsauce`. If your response variable (the one that you want to explain) is __y__, then:

- `Base` adjusts a linear regression to __y__, as a function of __X__ (optional) and __g(XW + b)__; without regularization of the regression coefficients. 
- `BayesianRVFL` (currently, `BayesianRVFLRegressor`) adds a ridge regularization parameter to the regression coefficients of `Base`, which prevents overfitting. Confidence intervals around the prediction can also be obtained.  
- `BayesianRVFL2` (currently, `BayesianRVFL2Regressor`) adds 2 regularization parameters to `Base`. As with `BayesianRVFL`, confidence intervals around the prediction can be obtained.
- `Custom` (`CustomRegressor` and `CustomClassifier`) works with any object `fit_obj` possessing methods `fit_obj.fit()` and `fit_obj.predict()`. Notably, the model can be applied to any [`scikit-learn`](https://scikit-learn.org)'s model, and to [`xgboost`](https://github.com/dmlc/xgboost), [`LightGBM`](https://github.com/Microsoft/LightGBM), [`CatBoost`](https://github.com/catboost/catboost) models. It adjusts `fit_obj` to __y__, as a function of __X__ (optional) and __g(XW + b)__. `Custom` objects can also be combined to form __deeper learning architectures__, as it will be shown in the next section. 
- `MTS` does multivariate time series forecasting. Like `Custom`, it works with any object `fit_obj` possessing methods `fit_obj.fit()` and `fit_obj.predict()`.


## Quick start

Here, we present examples of use of `Base`, `BayesianRVFL`, `BayesianRVFL2`, an example of `Custom` model using `scikit-learn`, and an example of `MTS` forecasting. We start by importing the packages and datasets necessary for the demo:

````python
import nnetsauce as ns
import numpy as np      
import matplotlib.pyplot as plt
from sklearn import datasets, metrics

diabetes = datasets.load_diabetes()
X = diabetes.data 
y = diabetes.target

breast_cancer = datasets.load_breast_cancer()
Z = breast_cancer.data
t = breast_cancer.target
````

`Base` (no regularization):

````python
# create object Base 

# X is not used directly, but only g(XW+b) ('direct_link' parameter)
# W is drawn from a deterministic Sobol sequence ('nodes_sim' parameter)
# b is equal to 0 ('bias' parameter)
# The activation function g is the hyperbolic tangent ('activation_name' parameter)
# The data in X is clustered: 2 clusters are obtained with k-means before fitting the model ('type_clust', 'n_clusters' parameters)

fit_obj = ns.BaseRegressor(n_hidden_features=100, 
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

`BayesianRVFL` (one regularization parameter):

````python
# create object BayesianRVFL  
# regularization is controlled by 's' and 'sigma'
# here, nodes_sim='halton' is used in the hidden layer
fit_obj = ns.BayesianRVFLRegressor(n_hidden_features=100,
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
plt.plot(x, fit_obj.predict(X[350:375,:]), color='blue')
plt.title('preds vs test set obs')
plt.xlabel('x')
plt.ylabel('preds')
plt.show()
````

`BayesianRVFL2` (two regularization parameters):

````python
# create object BayesianRVFL2 
# regularization is controlled by 's1', 's2' and 'sigma'
fit_obj = ns.BayesianRVFL2Regressor(n_hidden_features=100, 
                  direct_link=True,
                  activation_name='tanh', 
                  n_clusters=3, 
                  s1=0.5, s2=0.1, sigma=0.1)

# fit training set 
fit_obj.fit(X[0:350,:], y[0:350])

# predict on test set 
x = np.linspace(351, 375, num = 375-351+1)
plt.scatter(x = x, y = y[350:375], color='black')
plt.plot(x, fit_obj.predict(X[350:375,:]), color='blue')
plt.title('preds vs test set obs')
plt.xlabel('x')
plt.ylabel('preds')
plt.show()
````

`Custom` (using `scikit-learn`):

````python
from sklearn import linear_model, gaussian_process

regr = linear_model.BayesianRidge()
regr2 = linear_model.ElasticNet()
regr3 = gaussian_process.GaussianProcessClassifier()

# create object Custom 
fit_obj = ns.CustomRegressor(obj=regr, n_hidden_features=100, 
                    direct_link=False, bias=True,
                    activation_name='tanh', n_clusters=2)

fit_obj2 = ns.CustomRegressor(obj=regr2, n_hidden_features=500, 
                    direct_link=True, bias=False,
                    activation_name='relu', n_clusters=0)

fit_obj3 = ns.CustomClassifier(obj = regr3, n_hidden_features=100, 
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

We can also __combine `Custom` building blocks__. __In the following example, doing that increases the accuracy__, as new layers are added to the stack:

````python

# layer 1 (base layer) ----
layer1_regr = linear_model.BayesianRidge()
layer1_regr.fit(X[0:100,:], y[0:100])
# RMSE score
np.sqrt(metrics.mean_squared_error(y[100:125], layer1_regr.predict(X[100:125,:])))


# layer 2 using layer 1 ----
layer2_regr = ns.CustomRegressor(obj = layer1_regr, n_hidden_features=3, 
                        direct_link=True, bias=True, 
                        nodes_sim='sobol', activation_name='tanh', 
                        n_clusters=2)
layer2_regr.fit(X[0:100,:], y[0:100])

# RMSE score
np.sqrt(layer2_regr.score(X[100:125,:], y[100:125]))

# layer 3 using layer 2 ----
layer3_regr = ns.CustomRegressor(obj = layer2_regr, n_hidden_features=5, 
                        direct_link=True, bias=True, 
                        nodes_sim='hammersley', activation_name='sigmoid', 
                        n_clusters=2)
layer3_regr.fit(X[0:100,:], y[0:100])

# RMSE score
np.sqrt(layer3_regr.score(X[100:125,:], y[100:125]))

````

`MTS` (multivariate time series forecasting):

````python
np.random.seed(123)

M = np.random.rand(10, 3)
M[:,0] = 10*M[:,0]
M[:,2] = 25*M[:,2]
print(M)

regr4 = gaussian_process.GaussianProcessRegressor()
obj_MTS = ns.MTS(regr4, lags = 1, n_hidden_features=5, 
                 bias = False)
obj_MTS.fit(M)
print(obj_MTS.predict())


# With a deep stack of 'Custom' objects (from previous snippet)
obj_MTS2 = ns.MTS(layer3_regr, lags = 1, n_hidden_features=5, 
                 bias = False)
obj_MTS2.fit(M)
print(obj_MTS2.predict())


# Choosing different scalings for the input variables (first input
# of tuple 'type_scaling') , hidden layer (second input
# of tuple 'type_scaling'), and clustering (third input
# of tuple 'type_scaling'). 
# This is also available for models Base, Custom etc.

# 'minmax', 'minmax', 'std' scalings
regr6 = linear_model.BayesianRidge()
obj_MTS3 = ns.MTS(regr6, lags = 1, n_hidden_features=2, 
                 bias = True, type_scaling = ('minmax', 'minmax', 'std'))
obj_MTS3.fit(M)
print(obj_MTS3.predict())

# 'minmax', 'standardization', 'minmax' scalings
regr7 = linear_model.BayesianRidge()
obj_MTS4 = ns.MTS(regr6, lags = 1, n_hidden_features=2, 
                 bias = True, type_scaling = ('minmax', 'std', 'minmax'))
obj_MTS4.fit(M)
print(obj_MTS4.predict())

````

__There are certainly many other creative ways of combining these objects__, that you can (including **[tests](#Tests)**) [contribute](#Contributing) !

## Model validation

Every function from [sklearn.model_selection](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.model_selection) can be used 
 for this purpose. For example, the following code snippet does 3-fold cross validation on breast 
cancer data, and returns model accuracies: 

```python
from sklearn.model_selection import cross_val_score

cross_val_score(fit_obj3, X = Z, y = t, cv=3)
```

## Contributing

Your contributions are welcome, and valuable. Please, make sure to __read__ the [Code of Conduct](CONTRIBUTING.md) first.

In Pull Requests, let's strive to use [`black`](https://black.readthedocs.io/en/stable/) for formatting: 

```bash
pip install black
black --line-length=60 file_submitted_for_pr.py
```

A few things that we could explore are:

- Creating a great documentation on [readthedocs.org](https://readthedocs.org/) 
- Find other ways to combine `Custom` objects, using your fertile imagination (including [tests](#Tests))
- Better management of dates for MTS objects (including [tests](#Tests))
- Enrich the [tests](#Tests)
- Make `nnetsauce` available to `R` users 
- Any benchmarking of `nnetsauce` models (notebooks, files, etc.) can be stored in [demo](/nnetsauce/demo), with the following naming convention:  `yourgithubname_ddmmyy_shortdescriptionofdemo.[py|ipynb]`


## Tests

Tests for `nnetsauce`'s features are located [here](nnetsauce/tests). In order to run them and obtain tests' coverage (using [`nose2`](https://nose2.readthedocs.io/en/latest/)), do: 

- Install packages required for testing: 

```bash
pip install nose2
pip install coverage
```

- Run tests and print coverage:

```bash
git clone https://github.com/thierrymoudiki/nnetsauce.git
cd nnetsauce
nose2 --with-coverage
```

- Obtain coverage reports:

At the command line:

```bash
coverage report -m
```

  or an html report:

```bash
coverage html
```

## Dependencies 

- Numpy
- Scipy
- scikit-learn


## References

- Jones E, Oliphant E, Peterson P, et al. SciPy: Open Source Scientific Tools for Python, 2001-, http://www.scipy.org/ [Online; accessed 2019-01-04]
- Scikit-learn: Machine Learning in Python, Pedregosa et al., JMLR 12, pp. 2825-2830, 2011.


## License

[BSD 3-Clause](LICENSE) © Thierry Moudiki, 2019. 
