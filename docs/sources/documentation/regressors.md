# Regressors

_In alphabetical order_

All models possess methods `fit`, `predict`, and `score`. Methods `predict` and `score` are only documented for the first model; the __same principles apply subsequently__. For scoring metrics, refer to [scoring metrics](scoring_metrics.md). 

<span style="float:right;">[[source]](https://github.com/Techtonique/nnetsauce/blob/master/nnetsauce/base/baseRegressor.py#L14)</span>

### BaseRegressor


```python
nnetsauce.BaseRegressor(
    n_hidden_features=5,
    activation_name="relu",
    a=0.01,
    nodes_sim="sobol",
    bias=True,
    dropout=0,
    direct_link=True,
    n_clusters=2,
    cluster_encode=True,
    type_clust="kmeans",
    type_scaling=("std", "std", "std"),
    col_sample=1,
    row_sample=1,
    seed=123,
    backend="cpu",
)
```


Random Vector Functional Link Network regression without shrinkage

Parameters:

    n_hidden_features: int
        number of nodes in the hidden layer

    activation_name: str
        activation function: 'relu', 'tanh', 'sigmoid', 'prelu' or 'elu'

    a: float
        hyperparameter for 'prelu' or 'elu' activation function

    nodes_sim: str
        type of simulation for hidden layer nodes: 'sobol', 'hammersley', 'halton',
        'uniform'

    bias: boolean
        indicates if the hidden layer contains a bias term (True) or
        not (False)

    dropout: float
        regularization parameter; (random) percentage of nodes dropped out
        of the training

    direct_link: boolean
        indicates if the original features are included (True) in model's
        fitting or not (False)

    n_clusters: int
        number of clusters for type_clust='kmeans' or type_clust='gmm'
        clustering (could be 0: no clustering)

    cluster_encode: bool
        defines how the variable containing clusters is treated (default is one-hot);
        if `False`, then labels are used, without one-hot encoding

    type_clust: str
        type of clustering method: currently k-means ('kmeans') or Gaussian
        Mixture Model ('gmm')

    type_scaling: a tuple of 3 strings
        scaling methods for inputs, hidden layer, and clustering respectively
        (and when relevant).
        Currently available: standardization ('std') or MinMax scaling ('minmax')

    col_sample: float
        percentage of features randomly chosen for training

    row_sample: float
        percentage of rows chosen for training, by stratified bootstrapping

    seed: int
        reproducibility seed for nodes_sim=='uniform', clustering and dropout

    backend: str
        "cpu" or "gpu" or "tpu"

Attributes:

    beta_: vector
        regression coefficients

    GCV_: float
        Generalized Cross-Validation error


----

<span style="float:right;">[[source]](https://github.com/Techtonique/nnetsauce/blob/master/nnetsauce/base/baseRegressor.py#L122)</span>

### fit


```python
BaseRegressor.fit(X, y, **kwargs)
```


Fit BaseRegressor to training data (X, y)

Parameters:

    X: {array-like}, shape = [n_samples, n_features]
        Training vectors, where n_samples is the number
        of samples and n_features is the number of features

    y: array-like, shape = [n_samples]
        Target values

    **kwargs: additional parameters to be passed to self.cook_training_set

Returns:

    self: object


----

<span style="float:right;">[[source]](https://github.com/Techtonique/nnetsauce/blob/master/nnetsauce/base/baseRegressor.py#L153)</span>

### predict


```python
BaseRegressor.predict(X, **kwargs)
```


Predict test data X.

Parameters:

    X: {array-like}, shape = [n_samples, n_features]
        Training vectors, where n_samples is the number
        of samples and n_features is the number of features

    **kwargs: additional parameters to be passed to self.cook_test_set

Returns:

    model predictions: {array-like}


----

<span style="float:right;">[[source]](https://github.com/Techtonique/nnetsauce/blob/master/nnetsauce/base/baseRegressor.py#L192)</span>

### score


```python
BaseRegressor.score(X, y, scoring=None, **kwargs)
```


Score the model on test set features X and response y.

Parameters:

    X: {array-like}, shape = [n_samples, n_features]
        Training vectors, where n_samples is the number
        of samples and n_features is the number of features

    y: array-like, shape = [n_samples]
        Target values

    scoring: str
        must be in ('explained_variance', 'neg_mean_absolute_error',
                    'neg_mean_squared_error', 'neg_mean_squared_log_error',
                    'neg_median_absolute_error', 'r2')

    **kwargs: additional parameters to be passed to scoring functions

Returns:

model scores: {array-like}


----

<span style="float:right;">[[source]](https://github.com/Techtonique/nnetsauce/blob/master/nnetsauce/rvfl/bayesianrvflRegressor.py#L14)</span>

### BayesianRVFLRegressor


```python
nnetsauce.BayesianRVFLRegressor(
    n_hidden_features=5,
    activation_name="relu",
    a=0.01,
    nodes_sim="sobol",
    bias=True,
    dropout=0,
    direct_link=True,
    n_clusters=2,
    cluster_encode=True,
    type_clust="kmeans",
    type_scaling=("std", "std", "std"),
    seed=123,
    s=0.1,
    sigma=0.05,
    return_std=True,
    backend="cpu",
)
```


Bayesian Random Vector Functional Link Network regression with one prior

Parameters:

    n_hidden_features: int
        number of nodes in the hidden layer

    activation_name: str
        activation function: 'relu', 'tanh', 'sigmoid', 'prelu' or 'elu'

    a: float
        hyperparameter for 'prelu' or 'elu' activation function

    nodes_sim: str
        type of simulation for the nodes: 'sobol', 'hammersley', 'halton', 'uniform'

    bias: boolean
        indicates if the hidden layer contains a bias term (True) or not (False)

    dropout: float
        regularization parameter; (random) percentage of nodes dropped out
        of the training

    direct_link: boolean
        indicates if the original features are included (True) in model''s fitting or not (False)

    n_clusters: int
        number of clusters for 'kmeans' or 'gmm' clustering (could be 0: no clustering)

    cluster_encode: bool
        defines how the variable containing clusters is treated (default is one-hot)
        if `False`, then labels are used, without one-hot encoding

    type_clust: str
        type of clustering method: currently k-means ('kmeans') or Gaussian Mixture Model ('gmm')

    type_scaling: a tuple of 3 strings
        scaling methods for inputs, hidden layer, and clustering respectively
        (and when relevant).
        Currently available: standardization ('std') or MinMax scaling ('minmax')

    seed: int
        reproducibility seed for nodes_sim=='uniform'

    s: float
        std. dev. of regression parameters in Bayesian Ridge Regression

    sigma: float
        std. dev. of residuals in Bayesian Ridge Regression

    return_std: boolean
        if True, uncertainty around predictions is evaluated

    backend: str
        "cpu" or "gpu" or "tpu"

Attributes:

    beta_: array-like
        regression''s coefficients

    Sigma_: array-like
        covariance of the distribution of fitted parameters

    GCV_: float
        Generalized cross-validation error

    y_mean_: float
        average response

Examples:

```python
TBD
```


----

<span style="float:right;">[[source]](https://github.com/Techtonique/nnetsauce/blob/master/nnetsauce/rvfl/bayesianrvflRegressor.py#L137)</span>

### fit


```python
BayesianRVFLRegressor.fit(X, y, **kwargs)
```


Fit BayesianRVFLRegressor to training data (X, y).

Parameters:

    X: {array-like}, shape = [n_samples, n_features]
        Training vectors, where n_samples is the number
        of samples and n_features is the number of features.

    y: array-like, shape = [n_samples]
        Target values.

    **kwargs: additional parameters to be passed to
            self.cook_training_set

Returns:

    self: object


----

<span style="float:right;">[[source]](https://github.com/Techtonique/nnetsauce/blob/master/nnetsauce/rvfl/bayesianrvfl2Regressor.py#L14)</span>

### BayesianRVFL2Regressor


```python
nnetsauce.BayesianRVFL2Regressor(
    n_hidden_features=5,
    activation_name="relu",
    a=0.01,
    nodes_sim="sobol",
    bias=True,
    dropout=0,
    direct_link=True,
    n_clusters=0,
    cluster_encode=True,
    type_clust="kmeans",
    type_scaling=("std", "std", "std"),
    seed=123,
    s1=0.1,
    s2=0.1,
    sigma=0.05,
    return_std=True,
    backend="cpu",
)
```


Bayesian Random Vector Functional Link Network regression with two priors

Parameters:

    n_hidden_features: int
        number of nodes in the hidden layer

    activation_name: str
        activation function: 'relu', 'tanh', 'sigmoid', 'prelu' or 'elu'

    a: float
        hyperparameter for 'prelu' or 'elu' activation function

    nodes_sim: str
        type of simulation for the nodes: 'sobol', 'hammersley', 'halton', 'uniform'

    bias: boolean
        indicates if the hidden layer contains a bias term (True) or not (False)

    dropout: float
        regularization parameter; (random) percentage of nodes dropped out
        of the training

    direct_link: boolean
        indicates if the original features are included (True) in model''s fitting or not (False)

    n_clusters: int
        number of clusters for 'kmeans' or 'gmm' clustering (could be 0: no clustering)

    cluster_encode: bool
        defines how the variable containing clusters is treated (default is one-hot)
        if `False`, then labels are used, without one-hot encoding

    type_clust: str
        type of clustering method: currently k-means ('kmeans') or Gaussian Mixture Model ('gmm')

    type_scaling: a tuple of 3 strings
        scaling methods for inputs, hidden layer, and clustering respectively
        (and when relevant).
        Currently available: standardization ('std') or MinMax scaling ('minmax')

    seed: int
        reproducibility seed for nodes_sim=='uniform'

    s1: float
        std. dev. of init. regression parameters in Bayesian Ridge Regression

    s2: float
        std. dev. of augmented regression parameters in Bayesian Ridge Regression

    sigma: float
        std. dev. of residuals in Bayesian Ridge Regression

    return_std: boolean
        if True, uncertainty around predictions is evaluated

    backend: str
        "cpu" or "gpu" or "tpu"

Attributes:

    beta_: array-like
        regression''s coefficients

    Sigma_: array-like
        covariance of the distribution of fitted parameters

    GCV_: float
        Generalized cross-validation error

    y_mean_: float
        average response

Examples:

```python
TBD
```


----

<span style="float:right;">[[source]](https://github.com/Techtonique/nnetsauce/blob/master/nnetsauce/rvfl/bayesianrvfl2Regressor.py#L143)</span>

### fit


```python
BayesianRVFL2Regressor.fit(X, y, **kwargs)
```


Fit BayesianRVFL2Regressor to training data (X, y)

Parameters:

    X: {array-like}, shape = [n_samples, n_features]
        Training vectors, where n_samples is the number
        of samples and n_features is the number of features

    y: array-like, shape = [n_samples]
        Target values

    **kwargs: additional parameters to be passed to
            self.cook_training_set

Returns:

    self: object


----

<span style="float:right;">[[source]](https://github.com/Techtonique/nnetsauce/blob/master/nnetsauce/custom/customRegressor.py#L12)</span>

### CustomRegressor


```python
nnetsauce.CustomRegressor(
    obj,
    n_hidden_features=5,
    activation_name="relu",
    a=0.01,
    nodes_sim="sobol",
    bias=True,
    dropout=0,
    direct_link=True,
    n_clusters=2,
    cluster_encode=True,
    type_clust="kmeans",
    type_scaling=("std", "std", "std"),
    col_sample=1,
    row_sample=1,
    seed=123,
    backend="cpu",
)
```


Custom Regression model

This class is used to 'augment' any regression model with transformed features.

Parameters:

    obj: object
        any object containing a method fit (obj.fit()) and a method predict
        (obj.predict())

    n_hidden_features: int
        number of nodes in the hidden layer

    activation_name: str
        activation function: 'relu', 'tanh', 'sigmoid', 'prelu' or 'elu'

    a: float
        hyperparameter for 'prelu' or 'elu' activation function

    nodes_sim: str
        type of simulation for the nodes: 'sobol', 'hammersley', 'halton',
        'uniform'

    bias: boolean
        indicates if the hidden layer contains a bias term (True) or not
        (False)

    dropout: float
        regularization parameter; (random) percentage of nodes dropped out
        of the training

    direct_link: boolean
        indicates if the original predictors are included (True) in model's
        fitting or not (False)

    n_clusters: int
        number of clusters for 'kmeans' or 'gmm' clustering (could be 0:
            no clustering)

    cluster_encode: bool
        defines how the variable containing clusters is treated (default is one-hot)
        if `False`, then labels are used, without one-hot encoding

    type_clust: str
        type of clustering method: currently k-means ('kmeans') or Gaussian
        Mixture Model ('gmm')

    type_scaling: a tuple of 3 strings
        scaling methods for inputs, hidden layer, and clustering respectively
        (and when relevant).
        Currently available: standardization ('std') or MinMax scaling ('minmax')

    col_sample: float
        percentage of covariates randomly chosen for training

    row_sample: float
        percentage of rows chosen for training, by stratified bootstrapping

    seed: int
        reproducibility seed for nodes_sim=='uniform'

    type_fit: str
        'regression'

    backend: str
        "cpu" or "gpu" or "tpu"

Examples:

```python
TBD
```


----

<span style="float:right;">[[source]](https://github.com/Techtonique/nnetsauce/blob/master/nnetsauce/custom/customRegressor.py#L131)</span>

### fit


```python
CustomRegressor.fit(X, y, sample_weight=None, **kwargs)
```


Fit custom model to training data (X, y).

Parameters:

    X: {array-like}, shape = [n_samples, n_features]
        Training vectors, where n_samples is the number
        of samples and n_features is the number of features.

    y: array-like, shape = [n_samples]
        Target values.

    **kwargs: additional parameters to be passed to
        self.cook_training_set or self.obj.fit

Returns:

    self: object


----

<span style="float:right;">[[source]](https://github.com/Techtonique/nnetsauce/blob/master/nnetsauce/glm/glmRegressor.py#L18)</span>

### GLMRegressor


```python
nnetsauce.GLMRegressor(
    n_hidden_features=5,
    lambda1=0.01,
    alpha1=0.5,
    lambda2=0.01,
    alpha2=0.5,
    family="gaussian",
    activation_name="relu",
    a=0.01,
    nodes_sim="sobol",
    bias=True,
    dropout=0,
    direct_link=True,
    n_clusters=2,
    cluster_encode=True,
    type_clust="kmeans",
    type_scaling=("std", "std", "std"),
    optimizer=Optimizer(),
    seed=123,
)
```


Generalized 'linear' models using quasi-randomized networks (regression)

Attributes:

    n_hidden_features: int
        number of nodes in the hidden layer

    lambda1: float
        regularization parameter for GLM coefficients on original features

    alpha1: float
        controls compromize between l1 and l2 norm of GLM coefficients on original features

    lambda2: float
        regularization parameter for GLM coefficients on nonlinear features

    alpha2: float
        controls compromize between l1 and l2 norm of GLM coefficients on nonlinear features

    family: str
        "gaussian", "laplace" or "poisson" (for now)

    activation_name: str
        activation function: 'relu', 'tanh', 'sigmoid', 'prelu' or 'elu'

    a: float
        hyperparameter for 'prelu' or 'elu' activation function

    nodes_sim: str
        type of simulation for the nodes: 'sobol', 'hammersley', 'halton',
        'uniform'

    bias: boolean
        indicates if the hidden layer contains a bias term (True) or not
        (False)

    dropout: float
        regularization parameter; (random) percentage of nodes dropped out
        of the training

    direct_link: boolean
        indicates if the original predictors are included (True) in model's
        fitting or not (False)

    n_clusters: int
        number of clusters for 'kmeans' or 'gmm' clustering (could be 0:
            no clustering)

    cluster_encode: bool
        defines how the variable containing clusters is treated (default is one-hot)
        if `False`, then labels are used, without one-hot encoding

    type_clust: str
        type of clustering method: currently k-means ('kmeans') or Gaussian
        Mixture Model ('gmm')

    type_scaling: a tuple of 3 strings
        scaling methods for inputs, hidden layer, and clustering respectively
        (and when relevant).
        Currently available: standardization ('std') or MinMax scaling ('minmax')

    optimizer: object
        optimizer, from class nnetsauce.utils.Optimizer

    seed: int
        reproducibility seed for nodes_sim=='uniform'

Attributes:

    beta_: vector
        regression coefficients

Examples:

See [https://github.com/Techtonique/nnetsauce/blob/master/examples/glm_regression.py](https://github.com/Techtonique/nnetsauce/blob/master/examples/glm_regression.py)


----

<span style="float:right;">[[source]](https://github.com/Techtonique/nnetsauce/blob/master/nnetsauce/glm/glmRegressor.py#L184)</span>

### fit


```python
GLMRegressor.fit(
    X, y, learning_rate=0.01, decay=0.1, batch_prop=1, tolerance=1e-05, optimizer=None, verbose=0, **kwargs
)
```


Fit GLM model to training data (X, y).

Args:

    X: {array-like}, shape = [n_samples, n_features]
        Training vectors, where n_samples is the number
        of samples and n_features is the number of features.

    y: array-like, shape = [n_samples]
        Target values.

    **kwargs: additional parameters to be passed to
            self.cook_training_set or self.obj.fit

Returns:

    self: object


----

<span style="float:right;">[[source]](https://github.com/Techtonique/nnetsauce/blob/master/nnetsauce/randombag/_randomBagRegressor.py#L18)</span>

### RandomBagRegressor


```python
nnetsauce.RandomBagRegressor(
    obj,
    n_estimators=10,
    n_hidden_features=1,
    activation_name="relu",
    a=0.01,
    nodes_sim="sobol",
    bias=True,
    dropout=0,
    direct_link=False,
    n_clusters=2,
    cluster_encode=True,
    type_clust="kmeans",
    type_scaling=("std", "std", "std"),
    col_sample=1,
    row_sample=1,
    n_jobs=None,
    seed=123,
    verbose=1,
    backend="cpu",
)
```


Randomized 'Bagging' Regression model

Parameters:

    obj: object
        any object containing a method fit (obj.fit()) and a method predict
        (obj.predict())

    n_estimators: int
        number of boosting iterations

    n_hidden_features: int
        number of nodes in the hidden layer

    activation_name: str
        activation function: 'relu', 'tanh', 'sigmoid', 'prelu' or 'elu'

    a: float
        hyperparameter for 'prelu' or 'elu' activation function

    nodes_sim: str
        type of simulation for the nodes: 'sobol', 'hammersley', 'halton',
        'uniform'

    bias: boolean
        indicates if the hidden layer contains a bias term (True) or not
        (False)

    dropout: float
        regularization parameter; (random) percentage of nodes dropped out
        of the training

    direct_link: boolean
        indicates if the original predictors are included (True) in model''s
        fitting or not (False)

    n_clusters: int
        number of clusters for 'kmeans' or 'gmm' clustering (could be 0:
            no clustering)

    cluster_encode: bool
        defines how the variable containing clusters is treated (default is one-hot)
        if `False`, then labels are used, without one-hot encoding

    type_clust: str
        type of clustering method: currently k-means ('kmeans') or Gaussian
        Mixture Model ('gmm')

    type_scaling: a tuple of 3 strings
        scaling methods for inputs, hidden layer, and clustering respectively
        (and when relevant).
        Currently available: standardization ('std') or MinMax scaling ('minmax')

    col_sample: float
        percentage of covariates randomly chosen for training

    row_sample: float
        percentage of rows chosen for training, by stratified bootstrapping

    seed: int
        reproducibility seed for nodes_sim=='uniform'

    backend: str
        "cpu" or "gpu" or "tpu"

Attributes:

    voter_: dict
        dictionary containing all the fitted base-learners


Examples:

```python
import numpy as np
import nnetsauce as ns
from sklearn.datasets import fetch_california_housing
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

X, y = fetch_california_housing(return_X_y=True, as_frame=False)

# split data into training test and test set
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2, random_state=13)

# Requires further tuning
obj = DecisionTreeRegressor(max_depth=3, random_state=123)
obj2 = ns.RandomBagRegressor(obj=obj, direct_link=False,
                            n_estimators=50,
                            col_sample=0.9, row_sample=0.9,
                            dropout=0, n_clusters=0, verbose=1)

obj2.fit(X_train, y_train)

print(np.sqrt(obj2.score(X_test, y_test))) # RMSE

```


----

<span style="float:right;">[[source]](https://github.com/Techtonique/nnetsauce/blob/master/nnetsauce/randombag/_randomBagRegressor.py#L170)</span>

### fit


```python
RandomBagRegressor.fit(X, y, **kwargs)
```


Fit Random 'Bagging' model to training data (X, y).

Args:

    X: {array-like}, shape = [n_samples, n_features]
        Training vectors, where n_samples is the number
        of samples and n_features is the number of features.

    y: array-like, shape = [n_samples]
        Target values.

    **kwargs: additional parameters to be passed to
            self.cook_training_set or self.obj.fit

Returns:

    self: object


----

<span style="float:right;">[[source]](https://github.com/Techtonique/nnetsauce/blob/master/nnetsauce/ridge2/ridge2Regressor.py#L20)</span>

### Ridge2Regressor


```python
nnetsauce.Ridge2Regressor(
    n_hidden_features=5,
    activation_name="relu",
    a=0.01,
    nodes_sim="sobol",
    bias=True,
    dropout=0,
    n_clusters=2,
    cluster_encode=True,
    type_clust="kmeans",
    type_scaling=("std", "std", "std"),
    lambda1=0.1,
    lambda2=0.1,
    seed=123,
    backend="cpu",
)
```


Ridge regression with 2 regularization parameters derived from class Ridge

Parameters:

    n_hidden_features: int
        number of nodes in the hidden layer

    activation_name: str
        activation function: 'relu', 'tanh', 'sigmoid', 'prelu' or 'elu'

    a: float
        hyperparameter for 'prelu' or 'elu' activation function

    nodes_sim: str
        type of simulation for the nodes: 'sobol', 'hammersley', 'halton',
        'uniform'

    bias: boolean
        indicates if the hidden layer contains a bias term (True) or not
        (False)

    dropout: float
        regularization parameter; (random) percentage of nodes dropped out
        of the training

    n_clusters: int
        number of clusters for 'kmeans' or 'gmm' clustering (could be 0:
            no clustering)

    cluster_encode: bool
        defines how the variable containing clusters is treated (default is one-hot)
        if `False`, then labels are used, without one-hot encoding

    type_clust: str
        type of clustering method: currently k-means ('kmeans') or Gaussian
        Mixture Model ('gmm')

    type_scaling: a tuple of 3 strings
        scaling methods for inputs, hidden layer, and clustering respectively
        (and when relevant).
        Currently available: standardization ('std') or MinMax scaling ('minmax')

    lambda1: float
        regularization parameter on direct link

    lambda2: float
        regularization parameter on hidden layer

    seed: int
        reproducibility seed for nodes_sim=='uniform'

    backend: str
        'cpu' or 'gpu' or 'tpu'

Attributes:

    beta_: {array-like}
        regression coefficients

    y_mean_: float
        average response


----

<span style="float:right;">[[source]](https://github.com/Techtonique/nnetsauce/blob/master/nnetsauce/ridge2/ridge2Regressor.py#L124)</span>

### fit


```python
Ridge2Regressor.fit(X, y, **kwargs)
```


Fit Ridge model to training data (X, y).

Args:

    X: {array-like}, shape = [n_samples, n_features]
        Training vectors, where n_samples is the number
        of samples and n_features is the number of features.

    y: array-like, shape = [n_samples]
        Target values.

    **kwargs: additional parameters to be passed to
            self.cook_training_set or self.obj.fit

Returns:

    self: object


----

