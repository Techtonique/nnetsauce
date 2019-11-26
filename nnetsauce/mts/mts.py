"""MTS model"""

# Authors: Thierry Moudiki
#
# License: BSD 3

# ts objects with rpy2
# ts objects with rpy2
# ts objects with rpy2
# ts objects with rpy2
# ts objects with rpy2

import numpy as np
from ..base import Base
from scipy.stats import norm
from ..utils import matrixops as mo
from ..utils import timeseries as ts
import sklearn.metrics as skm2
import pickle


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
       seed: int 
           reproducibility seed for nodes_sim=='uniform'
       lags: int
           number of lags for the time series 
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
        col_sample=1,
        seed=123,
        lags=1,
        return_std=False,
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
            cluster_encode=cluster_encode,
            type_clust=type_clust,
            type_scaling=type_scaling,
            col_sample=col_sample,
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
        self.xreg = None
        self.regressors_scaler = None
        self.y_means = {}
        self.preds = None
        self.return_std = return_std
        self.preds_std = []
        self.row_sample = 1

    def fit(self, X, xreg=None, **kwargs):
        """Fit MTS model to training data (X, y).
        
        Parameters
        ----------
        X: {array-like}, shape = [n_samples, n_features]
            Training time series, where n_samples is the number 
            of samples and n_features is the number of features.
            X must be in increasing order (most recent observations last)
        xreg: {array-like}, shape = [n_samples, n_features_xreg]
              Additional regressors to be passed to obj
              xreg must be in increasing order (most recent observations last)
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

        self.fit_objs.clear()

        self.y_means.clear()

        mts_input = ts.create_train_inputs(
            X[::-1], self.lags
        )

        self.y = mts_input[0]

        self.X = mts_input[1]

        if xreg is not None:

            assert (
                xreg.shape[0] == n
            ), "'xreg' and 'X' must have the same number or observations"

            self.xreg = xreg

            xreg_input = ts.create_train_inputs(
                xreg[::-1], self.lags
            )

            dummy_y, scaled_Z = self.cook_training_set(
                y=np.repeat(1, n),
                X=mo.cbind(self.X, xreg_input[1]),
                **kwargs
            )

        else:  # xreg is None

            # avoids scaling X p times in the loop
            dummy_y, scaled_Z = self.cook_training_set(
                y=np.repeat(1, n), X=self.X, **kwargs
            )

        # loop on all the time series and adjust self.obj.fit
        for i in range(p):
            y_mean = np.mean(self.y[:, i])
            self.y_means[i] = y_mean
            self.fit_objs[i] = pickle.loads(
                pickle.dumps(
                    self.obj.fit(
                        scaled_Z,
                        self.y[:, i] - y_mean,
                        **kwargs
                    ),
                    -1,
                )
            )

        self.y_mean = None

        return self

    def predict(
        self, h=5, level=95, new_xreg=None, **kwargs
    ):
        """Predict on horizon h.
        
        Parameters
        ----------
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
               
        Returns
        -------
        model predictions for horizon = h: {array-like}
        """

        if self.xreg is not None:
            assert (
                new_xreg is not None
            ), "'new_xreg' must be provided"

        self.preds = None

        self.preds = pickle.loads(pickle.dumps(self.y, -1))

        n_features = self.n_series * self.lags

        self.preds_std = np.zeros(h)

        if "return_std" in kwargs:

            self.return_std = kwargs["return_std"]

            qt = norm.ppf(1 - 0.5 * (1 - level / 100))

        if (
            new_xreg is not None
        ):  # Additional regressors provided

            try:
                n_obs_xreg, n_features_xreg = new_xreg.shape
                assert (
                    n_features_xreg == self.xreg.shape[1]
                ), "check number of series provided for 'new_xreg' (compare with self.xreg.shape[1])"
            except:
                n_obs_xreg = new_xreg.shape  # one series

            assert (
                n_obs_xreg == h
            ), "please provide values of regressors 'new_xreg' for the whole horizon 'h'"

            n_features_xreg = n_features_xreg * self.lags

            n_features_total = n_features + n_features_xreg

            inv_new_xreg = mo.rbind(self.xreg, new_xreg)[
                ::-1
            ]

        # Loop on horizon h
        for i in range(h):

            new_obs = ts.reformat_response(
                self.preds, self.lags
            )

            if (
                new_xreg is not None
            ):  # Additional regressors provided

                new_obs_xreg = ts.reformat_response(
                    inv_new_xreg, self.lags
                )

                new_X = mo.rbind(
                    np.union1d(
                        new_obs.reshape(1, n_features),
                        new_obs_xreg.reshape(
                            1, n_features_xreg
                        ),
                    ),
                    np.ones(n_features_total).reshape(
                        1, n_features_total
                    ),
                )

            else:

                new_X = mo.rbind(
                    new_obs.reshape(1, n_features),
                    np.ones(n_features).reshape(
                        1, n_features
                    ),
                )

            cooked_new_X = self.cook_test_set(
                new_X, **kwargs
            )

            predicted_cooked_new_X = self.obj.predict(
                cooked_new_X, **kwargs
            )

            if (
                self.return_std == False
            ):  # std. dev. is not returned

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

            else:  # std. dev. is returned

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
            self.return_std == False
        ):  # the std. dev. is returned

            return self.preds

        else:

            self.preds_std = self.preds_std[::-1].reshape(
                h, 1
            )

            # reshape
            self.preds_std = np.repeat(
                self.preds_std, self.n_series
            ).reshape(-1, self.n_series)

            return (
                self.preds,
                self.preds_std,
                self.preds - qt * self.preds_std,
                self.preds + qt * self.preds_std,
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
        preds = self.predict(
            h=h, return_std=False, **kwargs
        )

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
