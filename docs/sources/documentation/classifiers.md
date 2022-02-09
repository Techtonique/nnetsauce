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

Parameters:

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

Attributes:

    alpha_: list
        AdaBoost coefficients alpha_m

    base_learners_: dict
        a dictionary containing the base learners  

Examples:

See also [https://github.com/Techtonique/nnetsauce/blob/master/examples/adaboost_classification.py](https://github.com/Techtonique/nnetsauce/blob/master/examples/adaboost_classification.py)

```python
import nnetsauce as ns
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from time import time

breast_cancer = load_breast_cancer()
Z = breast_cancer.data
t = breast_cancer.target
np.random.seed(123)
X_train, X_test, y_train, y_test = train_test_split(Z, t, test_size=0.2)

# SAMME.R
clf = LogisticRegression(solver='liblinear', multi_class = 'ovr', 
                        random_state=123)
fit_obj = ns.AdaBoostClassifier(clf, 
                                n_hidden_features=int(11.22338867), 
                                direct_link=True,
                                n_estimators=250, learning_rate=0.01126343,
                                col_sample=0.72684326, row_sample=0.86429443,
                                dropout=0.63078613, n_clusters=2,
                                type_clust="gmm",
                                verbose=1, seed = 123, 
                                method="SAMME.R")  

start = time() 
fit_obj.fit(X_train, y_train) 
print(f"Elapsed {time() - start}") 

start = time() 
print(fit_obj.score(X_test, y_test))
print(f"Elapsed {time() - start}") 

preds = fit_obj.predict(X_test)                        

print(fit_obj.score(X_test, y_test, scoring="roc_auc"))
print(metrics.classification_report(preds, y_test))

```              


----

<span style="float:right;">[[source]](https://github.com/Techtonique/nnetsauce/nnetsauce/boosting/adaBoostClassifier.py#L212)</span>

### fit


```python
AdaBoostClassifier.fit(X, y, sample_weight=None, **kwargs)
```


Fit Boosting model to training data (X, y).

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

<span style="float:right;">[[source]](https://github.com/Techtonique/nnetsauce/nnetsauce/boosting/adaBoostClassifier.py#L383)</span>

### predict


```python
AdaBoostClassifier.predict(X, **kwargs)
```


Predict test data X.

Parameters:

    X: {array-like}, shape = [n_samples, n_features]
        Training vectors, where n_samples is the number
        of samples and n_features is the number of features.

    **kwargs: additional parameters to be passed to
          self.cook_test_set

Returns:

    model predictions: {array-like}


----

<span style="float:right;">[[source]](https://github.com/Techtonique/nnetsauce/nnetsauce/boosting/adaBoostClassifier.py#L402)</span>

### predict_proba


```python
AdaBoostClassifier.predict_proba(X, **kwargs)
```


Predict probabilities for test data X.

Parameters:

    X: {array-like}, shape = [n_samples, n_features]
        Training vectors, where n_samples is the number
        of samples and n_features is the number of features.

    **kwargs: additional parameters to be passed to
          self.cook_test_set

Returns:

    probability estimates for test data: {array-like}


----

<span style="float:right;">[[source]](https://github.com/Techtonique/nnetsauce/nnetsauce/boosting/adaBoostClassifier.py#L481)</span>

### score


```python
AdaBoostClassifier.score(X, y, scoring=None, **kwargs)
```


Score the model on test set features X and response y.

Parameters:

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

Examples:

```python 
import nnetsauce as ns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
from time import time

digits = load_digits()
X = digits.data
y = digits.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=123)

# layer 1 (base layer) ----    
layer1_regr = RandomForestClassifier(n_estimators=10, random_state=123)

start = time() 

layer1_regr.fit(X_train, y_train)

# Accuracy in layer 1
print(layer1_regr.score(X_test, y_test))

# layer 2 using layer 1 ----
layer2_regr = ns.CustomClassifier(obj = layer1_regr, n_hidden_features=5, 
                        direct_link=True, bias=True, 
                        nodes_sim='uniform', activation_name='relu', 
                        n_clusters=2, seed=123)
layer2_regr.fit(X_train, y_train)

# Accuracy in layer 2
print(layer2_regr.score(X_test, y_test))

# layer 3 using layer 2 ----
layer3_regr = ns.CustomClassifier(obj = layer2_regr, n_hidden_features=10, 
                        direct_link=True, bias=True, dropout=0.7,
                        nodes_sim='uniform', activation_name='relu', 
                        n_clusters=2, seed=123)
layer3_regr.fit(X_train, y_train)

# Accuracy in layer 3
print(layer3_regr.score(X_test, y_test))

print(f"Elapsed {time() - start}")  
```        


----

<span style="float:right;">[[source]](https://github.com/Techtonique/nnetsauce/nnetsauce/custom/customClassifier.py#L168)</span>

### fit


```python
CustomClassifier.fit(X, y, sample_weight=None, **kwargs)
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

Parameters:

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

Attributes:

    beta_: vector
        regression coefficients

Examples:

See [https://github.com/Techtonique/nnetsauce/blob/master/examples/glm_classification.py](https://github.com/Techtonique/nnetsauce/blob/master/examples/glm_classification.py)    


----

<span style="float:right;">[[source]](https://github.com/Techtonique/nnetsauce/nnetsauce/glm/glmClassifier.py#L207)</span>

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

Parameters:

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

Attributes:

    fit_objs_: dict
        objects adjusted to each individual time series

    n_classes_: int
        number of classes for the classifier

Examples:

See also [https://github.com/Techtonique/nnetsauce/blob/master/examples/mtask_classification.py](https://github.com/Techtonique/nnetsauce/blob/master/examples/mtask_classification.py)

```python
import nnetsauce as ns
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from time import time

breast_cancer = load_breast_cancer()
Z = breast_cancer.data
t = breast_cancer.target

X_train, X_test, y_train, y_test = train_test_split(Z, t, test_size=0.2, 
                                                    random_state=123+2*10)

# Linear Regression is used 
regr = LinearRegression()
fit_obj = ns.MultitaskClassifier(regr, n_hidden_features=5, 
                            n_clusters=2, type_clust="gmm")

start = time()
fit_obj.fit(X_train, y_train)
print(f"Elapsed {time() - start}") 

print(fit_obj.score(X_test, y_test))
print(fit_obj.score(X_test, y_test, scoring="roc_auc"))

start = time()
preds = fit_obj.predict(X_test)
print(f"Elapsed {time() - start}") 
print(metrics.classification_report(preds, y_test))
```


----

<span style="float:right;">[[source]](https://github.com/Techtonique/nnetsauce/nnetsauce/multitask/multitaskClassifier.py#L170)</span>

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

Attributes:

    voter_: dict
        dictionary containing all the fitted base-learners


Examples:

See also [https://github.com/Techtonique/nnetsauce/blob/master/examples/randombag_classification.py](https://github.com/Techtonique/nnetsauce/blob/master/examples/randombag_classification.py)

```python
import nnetsauce as ns
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from time import time


breast_cancer = load_breast_cancer()
Z = breast_cancer.data
t = breast_cancer.target
np.random.seed(123)
X_train, X_test, y_train, y_test = train_test_split(Z, t, test_size=0.2)

# decision tree
clf = DecisionTreeClassifier(max_depth=2, random_state=123)
fit_obj = ns.RandomBagClassifier(clf, n_hidden_features=2,
                                direct_link=True,
                                n_estimators=100, 
                                col_sample=0.9, row_sample=0.9,
                                dropout=0.3, n_clusters=0, verbose=1)

start = time()
fit_obj.fit(X_train, y_train)
print(f"Elapsed {time() - start}") 

print(fit_obj.score(X_test, y_test))
print(fit_obj.score(X_test, y_test, scoring="roc_auc"))

start = time()
preds = fit_obj.predict(X_test)
print(f"Elapsed {time() - start}") 
print(metrics.classification_report(preds, y_test))    
```


----

<span style="float:right;">[[source]](https://github.com/Techtonique/nnetsauce/nnetsauce/randombag/_randomBagClassifier.py#L182)</span>

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

Examples:

See also [https://github.com/Techtonique/nnetsauce/blob/master/examples/ridge_classification.py](https://github.com/Techtonique/nnetsauce/blob/master/examples/ridge_classification.py)

```python
import nnetsauce as ns
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from time import time


breast_cancer = load_breast_cancer()
X = breast_cancer.data
y = breast_cancer.target

# split data into training test and test set
np.random.seed(123)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# create the model with nnetsauce
fit_obj = ns.Ridge2Classifier(lambda1 = 6.90185578e+04, 
                            lambda2 = 3.17392781e+02, 
                            n_hidden_features=95, 
                            n_clusters=2, 
                            dropout = 3.62817383e-01,
                            type_clust = "gmm")

# fit the model on training set
start = time()
fit_obj.fit(X_train, y_train)
print(f"Elapsed {time() - start}") 

# get the accuracy on test set
start = time()
print(fit_obj.score(X_test, y_test))
print(f"Elapsed {time() - start}") 

# get area under the curve on test set (auc)
print(fit_obj.score(X_test, y_test, scoring="roc_auc"))    
```


----

<span style="float:right;">[[source]](https://github.com/Techtonique/nnetsauce/nnetsauce/ridge2/ridge2Classifier.py#L295)</span>

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

Attributes:

    voter_: dict
        dictionary containing all the fitted base-learners


Examples:

See also [https://github.com/Techtonique/nnetsauce/blob/master/examples/randombag_classification.py](https://github.com/Techtonique/nnetsauce/blob/master/examples/randombag_classification.py)

```python
import nnetsauce as ns
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from time import time


breast_cancer = load_breast_cancer()
Z = breast_cancer.data
t = breast_cancer.target
np.random.seed(123)
X_train, X_test, y_train, y_test = train_test_split(Z, t, test_size=0.2)

# decision tree
clf = DecisionTreeClassifier(max_depth=2, random_state=123)
fit_obj = ns.RandomBagClassifier(clf, n_hidden_features=2,
                                direct_link=True,
                                n_estimators=100, 
                                col_sample=0.9, row_sample=0.9,
                                dropout=0.3, n_clusters=0, verbose=1)

start = time()
fit_obj.fit(X_train, y_train)
print(f"Elapsed {time() - start}") 

print(fit_obj.score(X_test, y_test))
print(fit_obj.score(X_test, y_test, scoring="roc_auc"))

start = time()
preds = fit_obj.predict(X_test)
print(f"Elapsed {time() - start}") 
print(metrics.classification_report(preds, y_test))    
```


----

<span style="float:right;">[[source]](https://github.com/Techtonique/nnetsauce/nnetsauce/randombag/_randomBagClassifier.py#L182)</span>

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

