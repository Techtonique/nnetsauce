# Classifiers 

_In alphabetical order_

All models possess methods `fit`, `predict`, `predict_proba`, and `score`. For scoring metrics, refer to [scoring metrics](scoring_metrics.md).

<span style="float:right;">[[source]](https://github.com/Techtonique/nnetsauce/nnetsauce/boosting/adaBoostClassifier.py#L18)</span>

### AdaBoostClassifier


```python
nnetsauce.AdaBoostClassifier(
    obj,
    n_estimators=10,
    learning_rate=0.1,
    n_hidden_features=1,
    reg_lambda=0,
    reg_alpha=0.5,
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
    seed=123,
    verbose=1,
    method="SAMME",
    backend="cpu",
)
```


AdaBoost Classification (SAMME) model class derived from class Boosting

Attributes:

    obj: object
        any object containing a method fit (obj.fit()) and a method predict
        (obj.predict())

    n_estimators: int
        number of boosting iterations

    learning_rate: float
        learning rate of the boosting procedure

    n_hidden_features: int
        number of nodes in the hidden layer

    reg_lambda: float
        regularization parameter for weights

    reg_alpha: float
        controls compromize between l1 and l2 norm of weights

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

    method: str
        type of Adaboost method, 'SAMME' (discrete) or 'SAMME.R' (real)

    backend: str
        "cpu" or "gpu" or "tpu"


----

<span style="float:right;">[[source]](https://github.com/Techtonique/nnetsauce/nnetsauce/boosting/adaBoostClassifier.py#L156)</span>

### fit


```python
AdaBoostClassifier.fit(X, y, sample_weight=None, **kwargs)
```


Fit Boosting model to training data (X, y).

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

<span style="float:right;">[[source]](https://github.com/Techtonique/nnetsauce/nnetsauce/boosting/adaBoostClassifier.py#L327)</span>

### predict


```python
AdaBoostClassifier.predict(X, **kwargs)
```


Predict test data X.

Args:

    X: {array-like}, shape = [n_samples, n_features]
        Training vectors, where n_samples is the number
        of samples and n_features is the number of features.

    **kwargs: additional parameters to be passed to
          self.cook_test_set

Returns:

    model predictions: {array-like}


----

<span style="float:right;">[[source]](https://github.com/Techtonique/nnetsauce/nnetsauce/boosting/adaBoostClassifier.py#L346)</span>

### predict_proba


```python
AdaBoostClassifier.predict_proba(X, **kwargs)
```


Predict probabilities for test data X.

Args:

    X: {array-like}, shape = [n_samples, n_features]
        Training vectors, where n_samples is the number
        of samples and n_features is the number of features.

    **kwargs: additional parameters to be passed to
          self.cook_test_set

Returns:

    probability estimates for test data: {array-like}


----

<span style="float:right;">[[source]](https://github.com/Techtonique/nnetsauce/nnetsauce/boosting/adaBoostClassifier.py#L425)</span>

### score


```python
AdaBoostClassifier.score(X, y, scoring=None, **kwargs)
```


Score the model on test set features X and response y.

Args:

    X: {array-like}, shape = [n_samples, n_features]
        Training vectors, where n_samples is the number
        of samples and n_features is the number of features

    y: array-like, shape = [n_samples]
        Target values

    scoring: str
        must be in ('accuracy', 'average_precision',
                   'brier_score_loss', 'f1', 'f1_micro',
                   'f1_macro', 'f1_weighted',  'f1_samples',
                   'neg_log_loss', 'precision', 'recall',
                   'roc_auc')

    **kwargs: additional parameters to be passed to scoring functions

Returns:

    model scores: {array-like}


----

<span style="float:right;">[[source]](https://github.com/Techtonique/nnetsauce/nnetsauce/custom/customClassifier.py#L12)</span>

### CustomClassifier


```python
nnetsauce.CustomClassifier(
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


Custom Classification model

Attributes:

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

    backend: str
        "cpu" or "gpu" or "tpu"


----

<span style="float:right;">[[source]](https://github.com/Techtonique/nnetsauce/nnetsauce/custom/customClassifier.py#L120)</span>

### fit


```python
CustomClassifier.fit(X, y, sample_weight=None, **kwargs)
```


Fit custom model to training data (X, y).

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

<span style="float:right;">[[source]](https://github.com/Techtonique/nnetsauce/nnetsauce/glm/glmClassifier.py#L18)</span>

### GLMClassifier


```python
nnetsauce.GLMClassifier(
    n_hidden_features=5,
    lambda1=0.01,
    alpha1=0.5,
    lambda2=0.01,
    alpha2=0.5,
    family="expit",
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


Generalized 'linear' models using quasi-randomized networks (classification)

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


----

<span style="float:right;">[[source]](https://github.com/Techtonique/nnetsauce/nnetsauce/glm/glmClassifier.py#L198)</span>

### fit


```python
GLMClassifier.fit(
    X, y, learning_rate=0.01, decay=0.1, batch_prop=1, tolerance=1e-05, optimizer=None, verbose=1, **kwargs
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

<span style="float:right;">[[source]](https://github.com/Techtonique/nnetsauce/nnetsauce/multitask/multitaskClassifier.py#L15)</span>

### MultitaskClassifier


```python
nnetsauce.MultitaskClassifier(
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


Multitask Classification model based on regression models, with shared covariates

Attributes:

    obj: object
        any object (must be a regression model) containing a method fit (obj.fit())
        and a method predict (obj.predict())

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

    backend: str
        "cpu" or "gpu" or "tpu"

References:

    [1] Moudiki, T. (2020). Quasi-randomized networks for regression and classification, with two shrinkage parameters. Available at:
    https://www.researchgate.net/publication/339512391_Quasi-randomized_networks_for_regression_and_classification_with_two_shrinkage_parameters


----

<span style="float:right;">[[source]](https://github.com/Techtonique/nnetsauce/nnetsauce/multitask/multitaskClassifier.py#L129)</span>

### fit


```python
MultitaskClassifier.fit(X, y, sample_weight=None, **kwargs)
```


Fit MultitaskClassifier to training data (X, y).

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

<span style="float:right;">[[source]](https://github.com/Techtonique/nnetsauce/nnetsauce/randombag/_randomBagClassifier.py#L18)</span>

### RandomBagClassifier


```python
nnetsauce.RandomBagClassifier(
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


Randomized 'Bagging' Classification model

Attributes:

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

    backend: str
        "cpu" or "gpu" or "tpu"


----

<span style="float:right;">[[source]](https://github.com/Techtonique/nnetsauce/nnetsauce/randombag/_randomBagClassifier.py#L136)</span>

### fit


```python
RandomBagClassifier.fit(X, y, **kwargs)
```


Fit Random 'Forest' model to training data (X, y).

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

<span style="float:right;">[[source]](https://github.com/Techtonique/nnetsauce/nnetsauce/ridge2/ridge2Classifier.py#L17)</span>

### Ridge2Classifier


```python
nnetsauce.Ridge2Classifier(
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
    lambda1=0.1,
    lambda2=0.1,
    seed=123,
    backend="cpu",
)
```


Multinomial logit classification with 2 regularization parameters

Attributes:

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

    lambda1: float
        regularization parameter on direct link

    lambda2: float
        regularization parameter on hidden layer

    seed: int
        reproducibility seed for nodes_sim=='uniform'

    backend: str
        "cpu" or "gpu" or "tpu"


References:

    - [1] Moudiki, T. (2020). Quasi-randomized networks for regression and classification, with two shrinkage parameters. Available at:
    https://www.researchgate.net/publication/339512391_Quasi-randomized_networks_for_regression_and_classification_with_two_shrinkage_parameters

    - [2] Moudiki, T. (2019). Multinomial logistic regression using quasi-randomized networks. Available at:
    https://www.researchgate.net/publication/334706878_Multinomial_logistic_regression_using_quasi-randomized_networks


----

<span style="float:right;">[[source]](https://github.com/Techtonique/nnetsauce/nnetsauce/ridge2/ridge2Classifier.py#L261)</span>

### fit


```python
Ridge2Classifier.fit(X, y, solver="L-BFGS-B", **kwargs)
```


Fit Ridge model to training data (X, y).

for beta: regression coeffs (beta11, ..., beta1p, ..., betaK1, ..., betaKp)
for K classes and p covariates.

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

<span style="float:right;">[[source]](https://github.com/Techtonique/nnetsauce/nnetsauce/ridge2/ridge2MultitaskClassifier.py#L20)</span>

### Ridge2MultitaskClassifier


```python
nnetsauce.Ridge2MultitaskClassifier(
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


Multitask Ridge classification with 2 regularization parameters

Attributes:

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
        "cpu" or "gpu" or "tpu"

References:

    - [1] Moudiki, T. (2020). Quasi-randomized networks for regression and classification, with two shrinkage parameters. Available at:
    https://www.researchgate.net/publication/339512391_Quasi-randomized_networks_for_regression_and_classification_with_two_shrinkage_parameters


----

<span style="float:right;">[[source]](https://github.com/Techtonique/nnetsauce/nnetsauce/ridge2/ridge2MultitaskClassifier.py#L121)</span>

### fit


```python
Ridge2MultitaskClassifier.fit(X, y, **kwargs)
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

<span style="float:right;">[[source]](https://github.com/Techtonique/nnetsauce/nnetsauce/randombag/_randomBagClassifier.py#L18)</span>

### RandomBagClassifier


```python
nnetsauce.RandomBagClassifier(
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


Randomized 'Bagging' Classification model

Attributes:

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

    backend: str
        "cpu" or "gpu" or "tpu"


----

<span style="float:right;">[[source]](https://github.com/Techtonique/nnetsauce/nnetsauce/randombag/_randomBagClassifier.py#L136)</span>

### fit


```python
RandomBagClassifier.fit(X, y, **kwargs)
```


Fit Random 'Forest' model to training data (X, y).

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

