# Authors: Thierry Moudiki
#
# License: BSD 3

import numpy as np
import sklearn.metrics as skm2
from .custom import Custom
from ..utils import matrixops as mo
from ..predictioninterval import PredictionInterval
from sklearn.base import RegressorMixin
from functools import partial
from scipy.stats import norm


class CustomRegressor(Custom, RegressorMixin):
    """Custom Regression model

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

        type_pi: str.
            type of prediction interval; currently "kde" (default).
            Used only in `self.predict`, for `self.replications` > 0 and `self.kernel`
            in ('gaussian', 'tophat'). Default is `None`.

        replications: int.
            number of replications (if needed) for predictive simulation.
            Used only in `self.predict`, for `self.kernel` in ('gaussian',
            'tophat') and `self.type_pi = 'kde'`. Default is `None`.

        kernel: str.
            the kernel to use for kernel density estimation (used for predictive
            simulation in `self.predict`, with `method='splitconformal'` and
            `type_pi = 'kde'`). Currently, either 'gaussian' or 'tophat'.

        type_split: str.
            Type of splitting for conformal prediction. None (default), or
            "random" (random split of data) or "sequential" (sequential split of data)

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

    See [https://thierrymoudiki.github.io/blog/2024/03/18/python/conformal-and-bayesian-regression](https://thierrymoudiki.github.io/blog/2024/03/18/python/conformal-and-bayesian-regression)

    """

    # construct the object -----

    def __init__(
        self,
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
        type_pi=None,
        replications=None,
        kernel=None,
        type_split=None,
        col_sample=1,
        row_sample=1,
        seed=123,
        backend="cpu",
    ):
        super().__init__(
            obj=obj,
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

        self.type_fit = "regression"
        self.type_pi = type_pi
        self.replications = replications
        self.kernel = kernel
        self.type_split = type_split

    def fit(self, X, y, sample_weight=None, **kwargs):
        """Fit custom model to training data (X, y).

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

        centered_y, scaled_Z = self.cook_training_set(y=y, X=X, **kwargs)

        # if sample_weights, else: (must use self.row_index)
        if sample_weight is not None:
            self.obj.fit(
                scaled_Z,
                centered_y,
                sample_weight=np.ravel(sample_weight, order="C")[
                    self.index_row
                ],
                **kwargs
            )

            return self

        self.obj.fit(scaled_Z, centered_y, **kwargs)

        self.X_ = X

        self.y_ = y

        return self

    def predict(self, X, level=95, method=None, **kwargs):
        """Predict test data X.

        Parameters:

            X: {array-like}, shape = [n_samples, n_features]
                Training vectors, where n_samples is the number
                of samples and n_features is the number of features.

            level: int
                Level of confidence (default = 95)

            method: str
                `None`, or 'splitconformal', 'localconformal'
                prediction (if you specify `return_pi = True`)

            **kwargs: additional parameters
                    `return_pi = True` for conformal prediction,
                    with `method` in ('splitconformal', 'localconformal')
                    or `return_std = True` for `self.obj` in
                    (`sklearn.linear_model.BayesianRidge`,
                    `sklearn.linear_model.ARDRegressor`,
                    `sklearn.gaussian_process.GaussianProcessRegressor`)`

        Returns:

            model predictions:
                an array if uncertainty quantification is not requested,
                  or a tuple if with prediction intervals and simulations
                  if `return_std = True` (mean, standard deviation,
                  lower and upper prediction interval) or `return_pi = True`
                  ()

        """

        if "return_std" in kwargs:

            alpha = 100 - level
            pi_multiplier = norm.ppf(1 - alpha / 200)

            if len(X.shape) == 1:

                n_features = X.shape[0]
                new_X = mo.rbind(
                    X.reshape(1, n_features),
                    np.ones(n_features).reshape(1, n_features),
                )

                mean_, std_ = self.obj.predict(
                    self.cook_test_set(new_X, **kwargs), return_std=True
                )[0]

                preds = self.y_mean_ + mean_
                lower = self.y_mean_ + (mean_ - pi_multiplier * std_)
                upper = self.y_mean_ + (mean_ + pi_multiplier * std_)

                return preds, std_, lower, upper

            # len(X.shape) > 1
            mean_, std_ = self.obj.predict(
                self.cook_test_set(X, **kwargs), return_std=True
            )

            preds = self.y_mean_ + mean_
            lower = self.y_mean_ + (mean_ - pi_multiplier * std_)
            upper = self.y_mean_ + (mean_ + pi_multiplier * std_)

            return preds, std_, lower, upper

        if "return_pi" in kwargs:
            assert method in (
                "splitconformal",
                "localconformal",
            ), "method must be in ('splitconformal', 'localconformal')"
            self.pi = PredictionInterval(
                obj=self,
                method=method,
                level=level,
                type_pi=self.type_pi,
                replications=self.replications,
                kernel=self.kernel,
            )
            self.pi.fit(self.X_, self.y_)
            self.X_ = None
            self.y_ = None
            preds = self.pi.predict(X, return_pi=True)
            return preds

        # "return_std" not in kwargs
        if len(X.shape) == 1:

            n_features = X.shape[0]
            new_X = mo.rbind(
                X.reshape(1, n_features),
                np.ones(n_features).reshape(1, n_features),
            )

            return (
                self.y_mean_
                + self.obj.predict(
                    self.cook_test_set(new_X, **kwargs), **kwargs
                )
            )[0]

        # len(X.shape) > 1
        return self.y_mean_ + self.obj.predict(
            self.cook_test_set(X, **kwargs), **kwargs
        )
