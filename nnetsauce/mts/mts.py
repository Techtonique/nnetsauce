"""MTS model"""

# Authors: Thierry Moudiki
#
# License: BSD 3


# for additional (deterministic) regressors,
# modify the reformatting function
# for additional (deterministic) regressors,
# modify the reformatting function
# for additional (deterministic) regressors,
# modify the reformatting function
# for additional (deterministic) regressors,
# modify the reformatting function
# for additional (deterministic) regressors,
# modify the reformatting function

# change: return_std = True must be in method predict
# change: return_std = True must be in method predict
# change: return_std = True must be in method predict
# change: return_std = True must be in method predict
# change: return_std = True must be in method predict

# ts objects with rpy2
# ts objects with rpy2
# ts objects with rpy2
# ts objects with rpy2
# ts objects with rpy2

import numpy as np
from ..base import Base
from ..utils import matrixops as mo
from ..utils import timeseries as ts
import sklearn.metrics as skm2


class MTS(Base):
    """MTS model class derived from class Base
    
       Parameters
       ----------
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
       type_clust: str
           type of clustering method: currently k-means ('kmeans') or Gaussian 
           Mixture Model ('gmm')
       type_scaling: a tuple of 3 strings
           scaling methods for inputs, hidden layer, and clustering respectively
           (and when relevant). 
           Currently available: standardization ('std') or MinMax scaling ('minmax')
       seed: int 
           reproducibility seed for nodes_sim=='uniform'
       lags: int
           number of lags for the time series 
       fit_objs: dict
           a dictionary containing objects fitted with obj, one for each time series
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
        type_clust="kmeans",
        type_scaling=("std", "std", "std"),
        seed=123,
        lags=1,
    ):

        assert (
            np.int(lags) == lags
        ), "parameter 'lags' should be an integer"

        super().__init__(
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
            seed=seed,
        )

        self.obj = obj
        self.n_series = None
        self.lags = lags
        self.fit_objs = {}
        self.y = (
            None
        )  # MTS responses (most recent observations first)
        self.X = None  # MTS lags
        self.y_means = []
        self.preds = None
        self.preds_std = []


    def fit(self, X, **kwargs):
        """Fit MTS model to training data (X, y).
        
        Parameters
        ----------
        X: {array-like}, shape = [n_samples, n_features]
            Training time series, where n_samples is the number 
            of samples and n_features is the number of features.
            X must be in increasing order (most recent observations last)
        
        **kwargs: additional parameters to be passed to 
                  self.cook_training_set
               
        Returns
        -------
        self: object
        """

        self.y = None

        self.X = None

        n, p = X.shape

        self.n_series = p

        self.y_means = np.zeros(p)

        mts_input = ts.create_train_inputs(
            X[::-1], self.lags
        )

        self.y = mts_input[0]

        self.X = mts_input[1]

        # avoids scaling X p times in the loop
        scaled_Z = self.cook_training_set(
            y=np.repeat(1, self.n_series),
            X=self.X,
            **kwargs
        )

        # loop on all the time series and adjust self.obj.fit
        for i in range(p):
            y_mean = np.mean(self.y[:, i])
            self.y_means[i] = y_mean
            self.fit_objs[i] = self.obj.fit(
                scaled_Z, self.y[:, i] - y_mean, **kwargs
            )

        self.y_mean = None

        return self

    def predict(self, h=5, level=95, **kwargs):
        """Predict on horizon h.
        
        Parameters
        ----------
        h: {integer}
            Forecasting horizon
        
        level: {integer}
            Level of confidence (if obj has option 'return_std' and the 
            posterior is gaussian)
        
        **kwargs: additional parameters to be passed to 
                  self.cook_test_set
               
        Returns
        -------
        model predictions for horizon = h: {array-like}
        """

        self.preds = None

        self.preds = self.y

        n_features = self.n_series * self.lags

        self.preds_std = np.zeros(h)

        for i in range(h):

            new_obs = ts.reformat_response(
                self.preds, self.lags
            )

            new_X = mo.rbind(
                new_obs.reshape(1, n_features),
                np.ones(n_features).reshape(1, n_features),
            )

            cooked_new_X = self.cook_test_set(
                new_X, **kwargs
            )

            predicted_cooked_new_X = self.obj.predict(
                cooked_new_X, **kwargs
            )

            if (
                isinstance(predicted_cooked_new_X, tuple)
                == False
            ):  # std. dev. is returned

                preds = np.array(
                    [
                        (
                            self.y_means[j]
                            + predicted_cooked_new_X[0]
                        )
                        for j in range(self.n_series)
                    ]
                )

                self.preds = mo.rbind(preds, self.preds)

            else:  # std. dev. is not returned

                preds = np.array(
                    [
                        (
                            self.y_means[j]
                            + predicted_cooked_new_X[0][0]
                        )
                        for j in range(self.n_series)
                    ]
                )

                self.preds = mo.rbind(preds, self.preds)

                self.preds_std[i] = predicted_cooked_new_X[
                    1
                ][0]

        # function's return

        self.preds = self.preds[0:h, :][::-1]

        if (
            isinstance(predicted_cooked_new_X, tuple)
            == False
        ):  # the std. dev. is returned

            return self.preds

        else:

            self.preds_std = self.preds_std[::-1].reshape(
                h, 1
            )

            return (
                self.preds,
                self.preds_std,
                self.preds - self.preds_std,
                self.preds + self.preds_std,
            )

    def score(
        self,
        X,
        training_index,
        testing_index,
        scoring=None,
        **kwargs
    ):
        """ Train on training_index, score on testing_index. """

        assert (
            bool(
                set(training_index).intersection(
                    set(testing_index)
                )
            )
            == False
        ), "Non-overlapping 'training_index' and 'testing_index' required"

        # Dimensions
        n, p = X.shape

        # Training and testing sets
        X_train = X[training_index, :]
        X_test = X[testing_index, :]

        # Horizon
        h = len(testing_index)
        assert (
            len(training_index) + h
        ) <= n, "Please check lengths of training and testing windows"

        # Fit and predict
        self.fit(X_train, **kwargs)
        preds = self.predict(h=h, **kwargs)

        if (
            type(preds) == tuple
        ):  # if there are std. devs in the predictions
            preds = preds[
                0
            ]  # take the mean prediction only

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

        return tuple(
            [
                scoring_options[scoring](
                    X_test[:, i], preds[:, i], **kwargs
                )
                for i in range(p)
            ]
        )
