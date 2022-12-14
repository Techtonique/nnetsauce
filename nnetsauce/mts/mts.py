# Authors: Thierry Moudiki
#
# License: BSD 3 Clear Clause

import numpy as np
import pandas as pd
from ..base import Base
from scipy.stats import norm
from ..utils import matrixops as mo
from ..utils import timeseries as ts
import sklearn.metrics as skm2
import pickle
from functools import partial


class MTS(Base):
    """Univariate and multivariate time series (MTS) forecasting with Quasi-Randomized networks (Work in progress /!\)

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
        lags=1,
        seed=123,
        backend="cpu",
    ):

        assert int(lags) == lags, "parameter 'lags' should be an integer"

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
            seed=seed,
            backend=backend,
        )

        self.obj = obj
        self.n_series = None
        self.lags = lags
        self.fit_objs_ = {}
        self.y_ = None  # MTS responses (most recent observations first)
        self.X_ = None  # MTS lags
        self.xreg_ = None
        self.y_means_ = {}
        self.preds_ = None
        self.preds_std_ = []
        self.return_std_ = None
        self.df_ = None

    def fit(self, X, xreg=None):
        """Fit MTS model to training data X, with optional regressors xreg

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
        """

        if isinstance(X, pd.DataFrame):
            self.df_ = X
            X = X.values

        try:
            # multivariate time series
            n, p = X.shape
        except:
            # univariate time series
            n = X.shape[0]
            p = 1

        rep_1_n = np.repeat(1, n)

        self.y_ = None
        self.X_ = None
        self.n_series = p
        self.fit_objs_.clear()
        self.y_means_.clear()

        if p > 1:
            # multivariate time series
            mts_input = ts.create_train_inputs(X[::-1], self.lags)
        else:
            # univariate time series
            mts_input = ts.create_train_inputs(
                X.reshape(-1, 1)[::-1], self.lags
            )

        self.y_ = mts_input[0]
        
        self.X_ = mts_input[1]

        if xreg is not None:

            if isinstance(xreg, pd.DataFrame):
                xreg = xreg.values

            assert (
                xreg.shape[0] == n
            ), "'xreg' and 'X' must have the same number of observations"

            self.xreg_ = xreg

            xreg_input = ts.create_train_inputs(xreg[::-1], self.lags)

            dummy_y, scaled_Z = self.cook_training_set(
                y=rep_1_n,
                X=mo.cbind(self.X_, xreg_input[1], backend=self.backend),
            )

        else:  # xreg is None

            # avoids scaling X p times in the loop
            dummy_y, scaled_Z = self.cook_training_set(y=rep_1_n, X=self.X_)

        # loop on all the time series and adjust self.obj.fit
        for i in range(p):
            y_mean = np.mean(self.y_[:, i])
            self.y_means_[i] = y_mean
            self.obj.fit(scaled_Z, self.y_[:, i] - y_mean)
            self.fit_objs_[i] = pickle.loads(pickle.dumps(self.obj, -1))

        return self

    def predict(self, h=5, level=95, new_xreg=None, **kwargs):
        """Forecast all the time series, h steps ahead

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
        """

        if self.df_ is not None:  # `fit` takes a data frame input
                output_dates, frequency = ts.compute_output_dates(self.df_, h)

        self.return_std_ = False

        self.preds_ = None  # do not remove (!)

        self.preds_ = pickle.loads(pickle.dumps(self.y_, -1))

        y_means_ = np.asarray([self.y_means_[i] for i in range(self.n_series)])

        n_features = self.n_series * self.lags # check that /!\

        if "return_std" in kwargs:
            self.return_std_ = True 
            self.preds_std_ = []

        if self.xreg_ is None: # no external regressors

            for i in range(h):

                new_obs = ts.reformat_response(self.preds_, self.lags)

                new_X = new_obs.reshape(1, n_features)

                cooked_new_X = self.cook_test_set(new_X, **kwargs)

                if "return_std" in kwargs:
                    self.preds_std_.append([np.asarray(self.fit_objs_[i].predict(cooked_new_X, return_std=True)[1]).item() for i in range(self.n_series)])

                predicted_cooked_new_X = np.asarray([np.asarray(self.fit_objs_[i].predict(
                    cooked_new_X)).item() for i in range(self.n_series)])

                preds = np.asarray(y_means_ + predicted_cooked_new_X)                                                                     
                
                self.preds_ = mo.rbind(preds, self.preds_)
        
        else: # if self.xreg_ is not None: # with external regressors
            
            assert new_xreg is not None, "'new_xreg' must be provided to predict()"

            if isinstance(new_xreg, pd.DataFrame):
                new_xreg = new_xreg.values

            try:
                n_obs_xreg, n_features_xreg = new_xreg.shape
                assert (
                    n_features_xreg == self.xreg_.shape[1]
                ), "check number of inputs provided for 'new_xreg' (compare with self.xreg_.shape[1])"
            except:
                n_obs_xreg = new_xreg.shape  # one series

            assert (
                n_obs_xreg == h
            ), "please provide values of regressors 'new_xreg' for the whole horizon 'h'"            

            n_features_xreg = n_features_xreg * self.lags

            inv_new_xreg = mo.rbind(self.xreg_, new_xreg)[::-1]   

            for i in range(h):

                new_obs = ts.reformat_response(self.preds_, self.lags)

                new_obs_xreg = ts.reformat_response(inv_new_xreg, self.lags)

                new_X = np.concatenate(                            
                            (new_obs, new_obs_xreg), axis=None
                        ).reshape(1, -1)
                    
                cooked_new_X = self.cook_test_set(new_X, **kwargs)

                if "return_std" in kwargs:
                    self.preds_std_.append([np.asarray(self.fit_objs_[i].predict(
                        cooked_new_X, return_std=True)[1]).item() for i in range(self.n_series)])

                predicted_cooked_new_X = np.asarray([np.asarray(self.fit_objs_[i].predict(
                    cooked_new_X)).item() for i in range(self.n_series)])

                preds = np.asarray(y_means_ + predicted_cooked_new_X)                                                                     
                
                self.preds_ = mo.rbind(preds, self.preds_)        
        
        # function's return
        if self.df_ is None:
            self.preds_ = self.preds_[0:h, :][::-1]
            if "return_std" not in kwargs:
                return self.preds_                          
            self.preds_std_ = np.asarray(self.preds_std_)
            return self.preds_, self.preds_std_

        # if self.df_ is not None (return data frames)
        self.preds_ = pd.DataFrame(
            self.preds_[0:h, :][::-1],
            columns=self.df_.columns,
            index=output_dates,
        )
        if "return_std" not in kwargs:
            return self.preds_
        self.preds_std_ = pd.DataFrame(
            np.asarray(self.preds_std_),
            columns=self.df_.columns,
            index=output_dates,
        )
        return self.preds_, self.preds_std_



    def score(self, X, training_index, testing_index, scoring=None, **kwargs):
        """ Train on training_index, score on testing_index. """

        assert (
            bool(set(training_index).intersection(set(testing_index))) == False
        ), "Non-overlapping 'training_index' and 'testing_index' required"

        # Dimensions
        try:
            # multivariate time series
            n, p = X.shape
        except:
            # univariate time series
            n = X.shape[0]
            p = 1

        # Training and testing sets
        if p > 1:
            X_train = X[training_index, :]
            X_test = X[testing_index, :]
        else:
            X_train = X[training_index]
            X_test = X[testing_index]

        # Horizon
        h = len(testing_index)
        assert (
            len(training_index) + h
        ) <= n, "Please check lengths of training and testing windows"

        # Fit and predict
        self.fit(X_train, **kwargs)
        preds = self.predict(h=h, **kwargs)

        if scoring is None:
            scoring = "neg_root_mean_squared_error"

        # check inputs
        assert scoring in (
            "explained_variance",
            "neg_mean_absolute_error",
            "neg_mean_squared_error",
            "neg_root_mean_squared_error",
            "neg_mean_squared_log_error",
            "neg_median_absolute_error",
            "r2",
        ), "'scoring' should be in ('explained_variance', 'neg_mean_absolute_error', \
                               'neg_mean_squared_error', 'neg_root_mean_squared_error', 'neg_mean_squared_log_error', \
                               'neg_median_absolute_error', 'r2')"

        scoring_options = {
            "explained_variance": skm2.explained_variance_score,
            "neg_mean_absolute_error": skm2.mean_absolute_error,
            "neg_mean_squared_error": skm2.mean_squared_error,
            "neg_root_mean_squared_error": lambda x, y: np.sqrt(
                skm2.mean_squared_error(x, y)
            ),
            "neg_mean_squared_log_error": skm2.mean_squared_log_error,
            "neg_median_absolute_error": skm2.median_absolute_error,
            "r2": skm2.r2_score,
        }

        if p > 1:
            return tuple(
                [
                    scoring_options[scoring](
                        X_test[:, i], preds[:, i], **kwargs
                    )
                    for i in range(p)
                ]
            )
        else:
            return scoring_options[scoring](X_test, preds)
