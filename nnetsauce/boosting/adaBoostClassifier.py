# Authors: Thierry Moudiki
#
# License: BSD 3

import copy
import numpy as np
import sklearn.metrics as skm2
from .bst import Boosting
from ..predictionset import PredictionSet
from ..custom import CustomClassifier
from ..utils import matrixops as mo
from ..utils import misc as mx
from ..utils import Progbar
from math import exp, log
from sklearn.base import ClassifierMixin
from scipy.special import xlogy, expit
import pickle


class AdaBoostClassifier(Boosting, ClassifierMixin):
    """AdaBoost Classification (SAMME) model class derived from class Boosting

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

        verbose: int
            0 for no output, 1 for a progress bar (default is 1)

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

    """

    # construct the object -----

    def __init__(
        self,
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
    ):
        self.type_fit = "classification"
        self.verbose = verbose
        self.method = method
        self.reg_lambda = reg_lambda
        self.reg_alpha = reg_alpha

        super().__init__(
            obj=obj,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            n_hidden_features=n_hidden_features,
            activation_name=activation_name,
            a=a,
            nodes_sim=nodes_sim,
            bias=bias,
            dropout=dropout,
            direct_link=direct_link,
            n_clusters=n_clusters,
            cluster_encode=cluster_encode,
            type_clust=type_clust,
            type_scaling=type_scaling,
            col_sample=col_sample,
            row_sample=row_sample,
            seed=seed,
            backend=backend,
        )

        self.alpha_ = []
        self.base_learners_ = dict.fromkeys(range(n_estimators))

    def fit(self, X, y, sample_weight=None, **kwargs):
        """Fit Boosting model to training data (X, y).

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
        """

        assert mx.is_factor(y), "y must contain only integers"

        assert self.method in (
            "SAMME",
            "SAMME.R",
        ), "`method` must be either 'SAMME' or 'SAMME.R'"

        assert (self.reg_lambda <= 1) & (
            self.reg_lambda >= 0
        ), "must have self.reg_lambda <= 1 &  self.reg_lambda >= 0"

        assert (self.reg_alpha <= 1) & (
            self.reg_alpha >= 0
        ), "must have self.reg_alpha <= 1 &  self.reg_alpha >= 0"

        # training
        n, p = X.shape
        self.n_classes = len(np.unique(y))
        self.classes_ = np.unique(y)  # for compatibility with sklearn
        self.n_classes_ = len(self.classes_)  # for compatibility with sklearn

        if sample_weight is None:
            w_m = np.repeat(1.0 / n, n)
        else:
            w_m = np.asarray(sample_weight)

        base_learner = CustomClassifier(
            self.obj,
            n_hidden_features=self.n_hidden_features,
            activation_name=self.activation_name,
            a=self.a,
            nodes_sim=self.nodes_sim,
            bias=self.bias,
            dropout=self.dropout,
            direct_link=self.direct_link,
            n_clusters=self.n_clusters,
            type_clust=self.type_clust,
            type_scaling=self.type_scaling,
            col_sample=self.col_sample,
            row_sample=self.row_sample,
            seed=self.seed,
        )

        if self.verbose == 1:
            pbar = Progbar(self.n_estimators)

        if self.method == "SAMME":
            err_m = 1e6
            err_bound = 1 - 1 / self.n_classes
            self.alpha_.append(1.0)
            x_range_n = range(n)

            for m in range(self.n_estimators):
                preds = base_learner.fit(
                    X, y, sample_weight=w_m.ravel(), **kwargs
                ).predict(X)

                self.base_learners_.update(
                    {m: pickle.loads(pickle.dumps(base_learner, -1))}
                )

                cond = [y[i] != preds[i] for i in x_range_n]

                err_m = max(
                    sum([elt[0] * elt[1] for elt in zip(cond, w_m)]),
                    2.220446049250313e-16,
                )  # sum(w_m) == 1

                if self.reg_lambda > 0:
                    err_m += self.reg_lambda * (
                        (1 - self.reg_alpha) * 0.5 * sum([x**2 for x in w_m])
                        + self.reg_alpha * sum([abs(x) for x in w_m])
                    )

                err_m = min(err_m, err_bound)

                alpha_m = self.learning_rate * log(
                    (self.n_classes - 1) * (1 - err_m) / err_m
                )

                self.alpha_.append(alpha_m)

                w_m_temp = [exp(alpha_m * cond[i]) for i in x_range_n]

                sum_w_m = sum(w_m_temp)

                w_m = np.asarray([w_m_temp[i] / sum_w_m for i in x_range_n])

                base_learner.set_params(seed=self.seed + (m + 1) * 1000)

                if self.verbose == 1:
                    pbar.update(m)

            if self.verbose == 1:
                pbar.update(self.n_estimators)

            self.n_estimators = len(self.base_learners_)
            self.classes_ = np.unique(y)

            return self

        if self.method == "SAMME.R":
            Y = mo.one_hot_encode2(y, self.n_classes)

            if sample_weight is None:
                w_m = np.repeat(1.0 / n, n)  # (N, 1)

            else:
                w_m = np.asarray(sample_weight)

            for m in range(self.n_estimators):
                probs = base_learner.fit(
                    X, y, sample_weight=w_m.ravel(), **kwargs
                ).predict_proba(X)

                np.clip(
                    a=probs, a_min=2.220446049250313e-16, a_max=1.0, out=probs
                )

                self.base_learners_.update(
                    {m: pickle.loads(pickle.dumps(base_learner, -1))}
                )

                w_m *= np.exp(
                    -1.0
                    * self.learning_rate
                    * (1.0 - 1.0 / self.n_classes)
                    * xlogy(Y, probs).sum(axis=1)
                )

                w_m /= np.sum(w_m)

                base_learner.set_params(seed=self.seed + (m + 1) * 1000)

                if self.verbose == 1:
                    pbar.update(m)

            if self.verbose == 1:
                pbar.update(self.n_estimators)

            self.n_estimators = len(self.base_learners_)
            self.classes_ = np.unique(y)

            return self

    def predict(self, X, **kwargs):
        """Predict test data X.

        Parameters:

            X: {array-like}, shape = [n_samples, n_features]
                Training vectors, where n_samples is the number
                of samples and n_features is the number of features.

            **kwargs: additional parameters to be passed to
                  self.cook_test_set

        Returns:

            model predictions: {array-like}
        """
        return self.predict_proba(X, **kwargs).argmax(axis=1)

    def predict_proba(self, X, **kwargs):
        """Predict probabilities for test data X.

        Parameters:

            X: {array-like}, shape = [n_samples, n_features]
                Training vectors, where n_samples is the number
                of samples and n_features is the number of features.

            **kwargs: additional parameters to be passed to
                  self.cook_test_set

        Returns:

            probability estimates for test data: {array-like}

        """

        n_iter = len(self.base_learners_)

        if self.method == "SAMME":
            ensemble_learner = np.zeros((X.shape[0], self.n_classes))

            # if self.verbose == 1:
            #    pbar = Progbar(n_iter)

            for idx, base_learner in self.base_learners_.items():
                preds = base_learner.predict(X, **kwargs)

                ensemble_learner += self.alpha_[idx] * mo.one_hot_encode2(
                    preds, self.n_classes
                )

                # if self.verbose == 1:
                #    pbar.update(idx)

            # if self.verbose == 1:
            #    pbar.update(n_iter)

            expit_ensemble_learner = expit(ensemble_learner)

            sum_ensemble = expit_ensemble_learner.sum(axis=1)

            return expit_ensemble_learner / sum_ensemble[:, None]

        # if self.method == "SAMME.R":
        ensemble_learner = 0

        # if self.verbose == 1:
        #    pbar = Progbar(n_iter)

        for idx, base_learner in self.base_learners_.items():
            probs = base_learner.predict_proba(X, **kwargs)

            np.clip(a=probs, a_min=2.220446049250313e-16, a_max=1.0, out=probs)

            log_preds_proba = np.log(probs)

            ensemble_learner += (
                log_preds_proba - log_preds_proba.mean(axis=1)[:, None]
            )

            # if self.verbose == 1:
            #    pbar.update(idx)

        ensemble_learner *= self.n_classes - 1

        # if self.verbose == 1:
        #    pbar.update(n_iter)

        expit_ensemble_learner = expit(ensemble_learner)

        sum_ensemble = expit_ensemble_learner.sum(axis=1)

        return expit_ensemble_learner / sum_ensemble[:, None]
