"""RNN Regressor"""

# Authors: Thierry Moudiki
#
# License: BSD 3

import numpy as np
import sklearn.metrics as skm2
from .rnn import RNN
from sklearn.base import RegressorMixin
from ..utils import Progbar


class RNNRegressor(RNN, RegressorMixin):
    """RNN Regression model class derived from class RNN
    
       Parameters
       ----------
       obj: object
           any object containing a method fit (obj.fit()) and a method predict 
           (obj.predict())
       alpha: float
           smoothing parameter
       window: int
           size of training window
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
    """

    # construct the object -----

    def __init__(
        self,
        obj,
        alpha=0.5,
        window=2,
        n_hidden_features=5,
        activation_name="relu",
        a=0.01,
        nodes_sim="sobol",
        bias=True,
        dropout=0,
        direct_link=True,
        n_clusters=2,
        type_clust="kmeans",
        type_scaling=("std", "std", "std"),
        col_sample=1,  # probably don't want to subsample here
        row_sample=1,  # probably don't want to subsample here
        seed=123,
    ):

        super().__init__(
            obj=obj,
            alpha=alpha,
            n_hidden_features=n_hidden_features,
            activation_name=activation_name,
            a=a,
            nodes_sim=nodes_sim,
            bias=bias,
            dropout=dropout,
            direct_link=direct_link,
            n_clusters=n_clusters,
            type_clust=type_clust,
            type_scaling=type_scaling,
            col_sample=col_sample,
            row_sample=row_sample,
            seed=seed,
        )

        self.type_fit = "regression"
        self.window = window

    def fit(
        self,
        inputs,
        targets=None,
        scoring=None,
        n_params=None,
        verbose=0,
    ):
        """ Train the model on multiple steps. """

        steps = inputs.shape[0]

        assert steps > 0, "inputs.shape[0] must be > 0"

        self.steps = steps

        if targets is not None:
            assert (
                steps == targets.shape[0]
            ), "'inputs' and 'targets' must contain the same number of steps"
            self.last_target = np.transpose(
                targets[(steps - self.window) : steps, :]
            )
        else:
            self.last_target = np.transpose(
                inputs[(steps - self.window) : steps, :]
            )

        # loss obtained by fitting on training set
        loss = 0

        if targets is not None:

            j = self.window - 1
            n_steps = steps - self.window + 1

            if verbose == 1:
                pbar = Progbar(target=n_steps)

            for i in range(n_steps):

                print("i= \n")
                print(i)
                batch_index = range(i, i + self.window)
                self.fit_step(
                    X=inputs[batch_index, :],
                    y=targets[j, :],
                )
                loss += self.score_step(
                    X=inputs[batch_index, :],
                    y=targets[j, :],
                    scoring=scoring,
                )

                if verbose == 1:
                    pbar.update(i)

                j += 1

            if verbose == 1:
                pbar.update(n_steps)

            return loss / n_steps

        # targets is None
        j = self.window
        n_steps = steps - self.window

        if verbose == 1:
            pbar = Progbar(target=n_steps)

        for i in range(n_steps):
            print("i= \n")
            print(i)
            batch_index = range(i, i + self.window)
            self.fit_step(
                X=inputs[batch_index, :], y=inputs[j, :]
            )
            loss += self.score_step(
                X=inputs[batch_index, :],
                y=inputs[j, :],
                scoring=scoring,
            )

            if verbose == 1:
                pbar.update(i)

            j += 1

        if verbose == 1:
            pbar.update(n_steps)

        return loss / n_steps

    def predict(
        self, h=5, level=95, new_xreg=None, **kwargs
    ):

        assert (
            self.steps > 0
        ), "method 'fit' must be called first"

        n_series = self.last_target.shape[0]

        n_res = h + self.window

        res = np.zeros((n_series, n_res))

        print("self.last_target")
        print(self.last_target)

        print("self.last_target.shape")
        print(self.last_target.shape)

        try:
            res[
                :, 0 : self.window
            ] = self.last_target.reshape(
                self.last_target.shape[0],
                self.last_target.shape[1],
            )
        except:
            res[
                :, 0 : self.window
            ] = self.last_target.reshape(-1, 1)

        if "return_std" not in kwargs:

            for i in range(self.window, self.window + h):

                res[:, i] = self.predict_step(
                    X=res[:, (i - self.window) : i],
                    **kwargs
                )

            return np.transpose(res)[(n_res - h) : n_res, :]

    # one step, on one input
    def fit_step(self, X, y, **kwargs):
        """Fit RNN model to training data (X, y).
        
        Parameters
        ----------
        X: {array-like}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number 
            of samples and n_features is the number of features.
        
        y: array-like, shape = [n_samples]
               Target values.
    
        **kwargs: additional parameters to be passed to 
                  self.cook_training_set or self.obj.fit
               
        Returns
        -------
        self: object
        """
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
        else:
            X = np.transpose(X)

        print("========== \n")
        print("X: \n")
        print(X)
        print("\n")
        print("X.shape: \n")
        print(X.shape)
        print("\n")
        print("y: \n")
        print(y)
        print("\n")

        # calls 'create_layer' from parent RNN: obtains centered_y, updates state H.
        # 'scaled_Z' is not used, but H
        centered_y, scaled_Z = self.cook_training_set(
            y=y, X=X, **kwargs
        )

        self.obj.fit(X=scaled_Z, y=centered_y, **kwargs)

        return self

    # one step, on one input
    def predict_step(self, X, **kwargs):
        """Predict test data X.
        
        Parameters
        ----------
        X: {array-like}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number 
            of samples and n_features is the number of features.
        
        **kwargs: additional parameters to be passed to 
                  self.cook_test_set
               
        Returns
        -------
        model predictions: {array-like}
        """

        if len(X.shape) == 1:
            X = X.reshape(-1, 1)

        if "return_std" not in kwargs:

            return self.y_mean + self.obj.predict(
                self.cook_test_set(X, **kwargs), **kwargs
            )

        preds = self.obj.predict(
            self.cook_test_set(X, **kwargs), **kwargs
        )

        return (self.y_mean + preds[0], preds[1])

    def score_step(self, X, y, scoring=None, **kwargs):
        """ Score the model on test set covariates X and response y. """

        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
        else:
            X = np.transpose(X)

        if "return_std" not in kwargs:

            preds = self.predict_step(X)

        else:

            preds = self.predict_step(X)[0]

        if (
            type(preds) == tuple
        ):  # if there are std. devs in the predictions
            preds = preds[0]

        if scoring is None:
            scoring = "neg_mean_squared_error"

        # check inputs
        assert scoring in (
            "explained_variance",
            "neg_mean_absolute_error",
            "neg_mean_squared_error",
            "neg_mean_squared_log_error",
            "neg_median_absolute_error",
            "r2",
        ), "'scoring' should be in ('explained_variance', 'neg_mean_absolute_error', \
                           'neg_mean_squared_error', 'neg_mean_squared_log_error', \
                           'neg_median_absolute_error', 'r2')"

        scoring_options = {
            "explained_variance": skm2.explained_variance_score,
            "neg_mean_absolute_error": skm2.mean_absolute_error,
            "neg_mean_squared_error": skm2.mean_squared_error,
            "neg_mean_squared_log_error": skm2.mean_squared_log_error,
            "neg_median_absolute_error": skm2.median_absolute_error,
            "r2": skm2.r2_score,
        }

        return scoring_options[scoring](y, preds, **kwargs)
