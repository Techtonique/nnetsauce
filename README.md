![nnetsauce logo](the-nnetsauce.png)

<hr>

This package does Statistical/Machine Learning, using advanced combinations of randomized and quasi-randomized _neural_ networks layers. It contains models for regression, classification, and time series forecasting.

![PyPI](https://img.shields.io/pypi/v/nnetsauce) [![PyPI - License](https://img.shields.io/pypi/l/nnetsauce)](https://github.com/thierrymoudiki/nnetsauce/blob/master/LICENSE) [![Downloads](https://pepy.tech/badge/nnetsauce)](https://pepy.tech/project/nnetsauce) 


## Contents 
 [Installation for Python and R](#installation-for-Python-and-R) |
 [Package description](#package-description) |
 [Quick start](#quick-start) |
 [Model Validation](#model-validation) |
 [Contributing](#Contributing) |
 [Tests](#Tests) |
 [Dependencies](#dependencies) |
 [Citing `nnetsauce`](#Citation) |
 [References](#References) |
 [License](#License) 


## Installation (for Python and R)

### Python 

- __1st method__: by using `pip` at the command line for the stable version

```bash
pip install nnetsauce
```


- __2nd method__: from Github, for the development version

```bash
pip install git+https://github.com/thierrymoudiki/nnetsauce.git
```

### R 

- __1st method__: From Github, in R console:

```r
library(devtools)
devtools::install_github("thierrymoudiki/nnetsauce/R-package")
library(nnetsauce)
```

__General rule for using the package in R__:  object accesses with `.`'s are replaced by `$`'s. See also [Quick start](#quick-start).



## Package description

Every model in the `nnetsauce` is based on the component __g(XW + b)__, where:

- __X__ is a matrix containing the explanatory variables and (optional) clustering information. The clustering methods available are _k-means_ and a _Gaussian Mixture Model_; they help in taking into account input data's heterogeneity.
- __W__ creates new, additional explanatory variables from __X__. It can be drawn from various random and quasirandom sequences.
- __b__ is an optional bias parameter.
- __g__ is an _activation function_ such as the hyperbolic tangent or the sigmoid function (among others), that renders the combination of explanatory variables (through __W__) nonlinear.  

The complete __API documentation__ is available [online](#), and you can read blog posts about `nnetsauce` [here](https://thierrymoudiki.github.io/blog/#QuasiRandomizedNN).

## Quick start

We present Python examples of use of `CustomRegressor` and  `CustomClassifier`, and an example of Multivariate Time Series `MTS` forecasting. Multiple examples of use can also be found in [demo](/nnetsauce/demo) (notebooks), and [examples](/examples) (flat files); where you can contribute yours too with the following naming convention:  `yourgithubname_ddmmyy_shortdescriptionofdemo.[py|ipynb|R|Rmd]`. __For R examples__, you can type the following command in R console:

```r
help("CustomRegressor")
```

And read section __Examples__ in the Help page displayed (any other class name from the [__API documentation__](#) can be used instead of `CustomRegressor`). 

We start by importing the packages and datasets necessary for our demo:

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

`Custom` (using `scikit-learn`):

````python
import nnetsauce as ns
import numpy as np      
import matplotlib.pyplot as plt

from sklearn import datasets, metrics
from sklearn import linear_model, gaussian_process

# load datasets
diabetes = datasets.load_diabetes()
X = diabetes.data 
y = diabetes.target

breast_cancer = datasets.load_breast_cancer()
Z = breast_cancer.data
t = breast_cancer.target

# Base models for nnetsauce's Custom class
regr = linear_model.BayesianRidge()
regr2 = linear_model.ElasticNet()
regr3 = gaussian_process.GaussianProcessClassifier()

# create object Custom 
fit_obj = ns.CustomRegressor(obj=regr, n_hidden_features=100, 
                    direct_link=False, bias=True,
                    activation_name='tanh', n_clusters=2)

fit_obj2 = ns.CustomRegressor(obj=regr2, n_hidden_features=5, 
                    direct_link=True, bias=False,
                    activation_name='relu', n_clusters=0)

fit_obj3 = ns.CustomClassifier(obj = regr3, n_hidden_features=5, 
                    direct_link=True, bias=True,
                    activation_name='relu', n_clusters=2)

# fit model 1 on training set
index_train_1 = range(350)
index_test_1 = range(350, 442)
fit_obj.fit(X[index_train_1,:], y[index_train_1])

# predict on test set 
print(fit_obj.predict(X[index_test_1,:]))

# fit model 2 on training set
fit_obj2.fit(X[index_train_1,:], y[index_train_1])

# predict on test set 
print(fit_obj2.predict(X[index_test_1,:]))

# fit model 3 on training set
index_train_2 = range(455)
index_test_2 = range(455, 569)
fit_obj3.fit(Z[index_train_2,:], t[index_train_2])

# accuracy on test set 
print(fit_obj3.score(Z[index_test_2,:], t[index_test_2]))
````

We can __combine `Custom` building blocks__. __In the following example, doing that increases model accuracy__, as new layers are added to the stack:

````python

index_train = range(100)
index_test = range(100, 125)

# layer 1 (base layer) ----
layer1_regr = linear_model.BayesianRidge()
layer1_regr.fit(X[index_train,:], y[index_train])

# RMSE score on test set
print(np.sqrt(metrics.mean_squared_error(y[index_test], layer1_regr.predict(X[index_test,:]))))


# layer 2 using layer 1 ----
layer2_regr = ns.CustomRegressor(obj = layer1_regr, n_hidden_features=3, 
                        direct_link=True, bias=True, 
                        nodes_sim='sobol', activation_name='tanh', 
                        n_clusters=2)
layer2_regr.fit(X[index_train,:], y[index_train])

# RMSE score on test set
print(np.sqrt(layer2_regr.score(X[index_test,:], y[index_test])))

# layer 3 using layer 2 ----
layer3_regr = ns.CustomRegressor(obj = layer2_regr, n_hidden_features=5, 
                        direct_link=True, bias=True, 
                        nodes_sim='hammersley', activation_name='sigmoid', 
                        n_clusters=2)
layer3_regr.fit(X[index_test,:], y[index_test])

# RMSE score on test set
print(np.sqrt(layer3_regr.score(X[index_test,:], y[index_test])))

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
# This is also available for models Base, Custom, etc.

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

Your contributions are welcome, and valuable. Please, make sure to __read__ the [Code of Conduct](CONTRIBUTING.md) first. If you're not comfortable with Git/Version Control yet, please use [this form](https://forms.gle/HQVbrUsvZE7o8xco8) to provide a feedback.

In Pull Requests, let's strive to use [`black`](https://black.readthedocs.io/en/stable/) for formatting files: 

```bash
pip install black
black --line-length=80 file_submitted_for_pr.py
```

A few things that we could explore are:

- Creating a great documentation on [readthedocs.org](https://readthedocs.org/) --> [here](./docs) 
- Find other ways to combine `Custom` objects, using your fertile imagination (including [tests](#Tests))
- Better management of dates for MTS objects (including [tests](#Tests))
- Enrich the [tests](#Tests)
- Make `nnetsauce` available to `R` users --> [here](./R-package)
- Any benchmarking of `nnetsauce` models can be stored in [demo](/nnetsauce/demo) (notebooks) or [examples](./examples) (flat files), with the following naming convention:  `yourgithubname_ddmmyy_shortdescriptionofdemo.[py|ipynb|R|Rmd]`


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

## Citation

@misc{moudiki2019nnetsauce,
author={Moudiki, Thierry},
title={\code{nnetsauce}, {A} general-purpose tool for {S}tatistical/{M}achine Learning},
howpublished={\url{https://github.com/thierrymoudiki/nnetsauce}},
note={BSD 3-Clause Clear License. Version 0.x.x.},
year={2019--2020}
}

## References

- Moudiki, T. (2019). Multinomial logistic regression using quasi-randomized networks. Available at: https://www.researchgate.net/publication/334706878_Multinomial_logistic_regression_using_quasi-randomized_networks
- Moudiki  T,  Planchet  F,  Cousin  A  (2018).   “Multiple  Time  Series  Forecasting Using  Quasi-Randomized  Functional  Link  Neural  Networks. ”Risks, 6(1), 22. Available at: https://www.mdpi.com/2227-9091/6/1/22
- Jones E, Oliphant E, Peterson P, et al. SciPy: Open Source Scientific Tools for Python, 2001-, http://www.scipy.org/ [Online; accessed 2019-01-04]
- Scikit-learn: Machine Learning in Python, Pedregosa et al., JMLR 12, pp. 2825-2830, 2011.


## License

[BSD 3-Clause](LICENSE) © Thierry Moudiki, 2019. 
