# Time series models 

_In alphabetical order_

All models possess methods: `fit`, `predict`.

<span style="float:right;">[[source]](https://github.com/Techtonique/nnetsauce/nnetsauce/mts/mts.py#L16)</span>

### MTS


```python
nnetsauce.MTS(
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
    lags=1,
    seed=123,
    backend="cpu",
)
```


Univariate and multivariate time series (MTS) forecasting with Quasi-Randomized networks

Parameters:

    obj: object.
        any object containing a method fit (obj.fit()) and a method predict
        (obj.predict()).

    n_hidden_features: int.
        number of nodes in the hidden layer.

    activation_name: str.
        activation function: 'relu', 'tanh', 'sigmoid', 'prelu' or 'elu'.

    a: float.
        hyperparameter for 'prelu' or 'elu' activation function.

    nodes_sim: str.
        type of simulation for the nodes: 'sobol', 'hammersley', 'halton',
        'uniform'.

    bias: boolean.
        indicates if the hidden layer contains a bias term (True) or not
        (False).

    dropout: float.
        regularization parameter; (random) percentage of nodes dropped out
        of the training.

    direct_link: boolean.
        indicates if the original predictors are included (True) in model's fitting or not (False).

    n_clusters: int.
        number of clusters for 'kmeans' or 'gmm' clustering (could be 0: no clustering).

    cluster_encode: bool.
        defines how the variable containing clusters is treated (default is one-hot)
        if `False`, then labels are used, without one-hot encoding.

    type_clust: str.
        type of clustering method: currently k-means ('kmeans') or Gaussian
        Mixture Model ('gmm').

    type_scaling: a tuple of 3 strings.
        scaling methods for inputs, hidden layer, and clustering respectively
        (and when relevant).
        Currently available: standardization ('std') or MinMax scaling ('minmax').

    lags: int.
        number of lags used for each time series.

    seed: int.
        reproducibility seed for nodes_sim=='uniform'.

    backend: str.
        "cpu" or "gpu" or "tpu".

Attributes:

    fit_objs_: dict
        objects adjusted to each individual time series

    y_: {array-like}
        MTS responses (most recent observations first)

    X_: {array-like}
        MTS lags

    xreg_: {array-like}
        external regressors

    y_means_: dict
        a dictionary of each series mean values

    preds_: {array-like}
        successive model predictions

    preds_std_: {array-like}
        standard deviation around the predictions

    return_std_: boolean
        return uncertainty or not (set in predict)

    df_: data frame
        the input data frame, in case a data.frame is provided to `fit`

Examples:

Example 1:

```python
import nnetsauce as ns
import numpy as np
from sklearn import linear_model
np.random.seed(123)

M = np.random.rand(10, 3)
M[:,0] = 10*M[:,0]
M[:,2] = 25*M[:,2]
print(M)

# Adjust Bayesian Ridge
regr4 = linear_model.BayesianRidge()
obj_MTS = ns.MTS(regr4, lags = 1, n_hidden_features=5)
obj_MTS.fit(M)
print(obj_MTS.predict())

# with credible intervals
print(obj_MTS.predict(return_std=True, level=80))

print(obj_MTS.predict(return_std=True, level=95))
```

Example 2:

```python
import nnetsauce as ns
import numpy as np
from sklearn import linear_model

dataset = {
'date' : ['2001-01-01', '2002-01-01', '2003-01-01', '2004-01-01', '2005-01-01'],
'series1' : [34, 30, 35.6, 33.3, 38.1],
'series2' : [4, 5.5, 5.6, 6.3, 5.1],
'series3' : [100, 100.5, 100.6, 100.2, 100.1]}
df = pd.DataFrame(dataset).set_index('date')
print(df)

# Adjust Bayesian Ridge
regr5 = linear_model.BayesianRidge()
obj_MTS = ns.MTS(regr5, lags = 1, n_hidden_features=5)
obj_MTS.fit(df)
print(obj_MTS.predict())

# with credible intervals
print(obj_MTS.predict(return_std=True, level=80))

print(obj_MTS.predict(return_std=True, level=95))
```


----

<span style="float:right;">[[source]](https://github.com/Techtonique/nnetsauce/nnetsauce/mts/mts.py#L211)</span>

### fit


```python
MTS.fit(X, xreg=None)
```


Fit MTS model to training data X, with optional regressors xreg

Parameters:

    X: {array-like}, shape = [n_samples, n_features]
        Training time series, where n_samples is the number
        of samples and n_features is the number of features;
        X must be in increasing order (most recent observations last)

    xreg: {array-like}, shape = [n_samples, n_features_xreg]
        Additional regressors to be passed to obj
        xreg must be in increasing order (most recent observations last)

    **kwargs: additional parameters to be passed to
            self.cook_training_set

Returns:

    self: object


----

<span style="float:right;">[[source]](https://github.com/Techtonique/nnetsauce/nnetsauce/mts/mts.py#L296)</span>

### predict


```python
MTS.predict(h=5, level=95, new_xreg=None, **kwargs)
```


Forecast all the time series, h steps ahead

Parameters:

    h: {integer}
        Forecasting horizon

    level: {integer}
        Level of confidence (if obj has option 'return_std' and the
        posterior is gaussian)

    new_xreg: {array-like}, shape = [n_samples = h, n_new_xreg]
        New values of additional (deterministic) regressors on horizon = h
        new_xreg must be in increasing order (most recent observations last)

    **kwargs: additional parameters to be passed to
            self.cook_test_set

Returns:

    model predictions for horizon = h: {array-like}, data frame or tuple.
    Standard deviation and prediction intervals are returned when
    `obj.predict` can return standard deviation


----

