# Authors: T. Moudiki
#
# License: BSD 3 Clear Clause

import copy
import numpy as np
import pandas as pd
import sklearn.metrics as skm2
import matplotlib.pyplot as plt
import warnings
from collections import namedtuple
from copy import deepcopy
from functools import partial
from scipy.stats import describe, norm
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_pinball_loss
from tqdm import tqdm
from ..base import Base
from ..sampling import vinecopula_sample
from ..simulation import getsims, getsimsxreg
from ..quantile import QuantileRegressor
from ..utils import matrixops as mo
from ..utils import misc as mx
from ..utils import timeseries as ts
from ..utils import convert_df_to_numeric, coverage, winkler_score, mean_errors
from ..utils import TimeSeriesSplit


class MTS(Base):
    """Univariate and multivariate time series (MTS) forecasting with Quasi-Randomized networks

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
            If string, lags must be one of 'AIC', 'AICc', or 'BIC'.

        type_pi: str.
            type of prediction interval; currently:
            - "gaussian": simple, fast, but: assumes stationarity of Gaussian in-sample residuals and independence in the multivariate case
            - "quantile": use model-agnostic quantile regression under the hood
            - "kde": based on Kernel Density Estimation of in-sample residuals
            - "bootstrap": based on independent bootstrap of in-sample residuals
            - "block-bootstrap": based on basic block bootstrap of in-sample residuals
            - "scp-kde": Sequential split conformal prediction with Kernel Density Estimation of calibrated residuals
            - "scp-bootstrap": Sequential split conformal prediction with independent bootstrap of calibrated residuals
            - "scp-block-bootstrap": Sequential split conformal prediction with basic block bootstrap of calibrated residuals
            - "scp2-kde": Sequential split conformal prediction with Kernel Density Estimation of standardized calibrated residuals
            - "scp2-bootstrap": Sequential split conformal prediction with independent bootstrap of standardized calibrated residuals
            - "scp2-block-bootstrap": Sequential split conformal prediction with basic block bootstrap of standardized calibrated residuals
            - based on copulas of in-sample residuals: 'vine-tll', 'vine-bb1', 'vine-bb6', 'vine-bb7', 'vine-bb8', 'vine-clayton',
            'vine-frank', 'vine-gaussian', 'vine-gumbel', 'vine-indep', 'vine-joe', 'vine-student'
            - 'scp-vine-tll', 'scp-vine-bb1', 'scp-vine-bb6', 'scp-vine-bb7', 'scp-vine-bb8', 'scp-vine-clayton',
            'scp-vine-frank', 'scp-vine-gaussian', 'scp-vine-gumbel', 'scp-vine-indep', 'scp-vine-joe', 'scp-vine-student'
            - 'scp2-vine-tll', 'scp2-vine-bb1', 'scp2-vine-bb6', 'scp2-vine-bb7', 'scp2-vine-bb8', 'scp2-vine-clayton',
            'scp2-vine-frank', 'scp2-vine-gaussian', 'scp2-vine-gumbel', 'scp2-vine-indep', 'scp2-vine-joe', 'scp2-vine-student'

        level: int.
            level of confidence for `type_pi == 'quantile'` (default is `95`)

        block_size: int.
            size of block for 'type_pi' in ("block-bootstrap", "scp-block-bootstrap", "scp2-block-bootstrap").
            Default is round(3.15*(n_residuals^1/3))

        replications: int.
            number of replications (if needed, for predictive simulation). Default is 'None'.

        kernel: str.
            the kernel to use for residuals density estimation (used for predictive simulation). Currently, either 'gaussian' or 'tophat'.

        agg: str.
            either "mean" or "median" for simulation of bootstrap aggregating

        seed: int.
            reproducibility seed for nodes_sim=='uniform' or predictive simulation.

        backend: str.
            "cpu" or "gpu" or "tpu".

        verbose: int.
            0: not printing; 1: printing

        show_progress: bool.
            True: progress bar when fitting each series; False: no progress bar when fitting each series

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
            standard deviation around the predictions for Bayesian base learners (`obj`)

        gaussian_preds_std_: {array-like}
            standard deviation around the predictions for `type_pi='gaussian'`

        return_std_: boolean
            return uncertainty or not (set in predict)

        df_: data frame
            the input data frame, in case a data.frame is provided to `fit`

        n_obs_: int
            number of time series observations (number of rows for multivariate)

        level_: int
            level of confidence for prediction intervals (default is 95)

        residuals_: {array-like}
            in-sample residuals (for `type_pi` not conformal prediction) or calibrated residuals
            (for `type_pi` in conformal prediction)

        residuals_sims_: tuple of {array-like}
            simulations of in-sample residuals (for `type_pi` not conformal prediction) or
            calibrated residuals (for `type_pi` in conformal prediction)

        kde_: A scikit-learn object, see https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KernelDensity.html

        residuals_std_dev_: residuals standard deviation

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
        type_pi="kde",
        level=95,
        block_size=None,
        replications=None,
        kernel="gaussian",
        agg="mean",
        seed=123,
        backend="cpu",
        verbose=0,
        show_progress=True,
    ):
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

        # Add validation for lags parameter
        if isinstance(lags, str):
            assert lags in (
                "AIC",
                "AICc",
                "BIC",
            ), "if string, lags must be one of 'AIC', 'AICc', or 'BIC'"
        else:
            assert (
                int(lags) == lags
            ), "if numeric, lags parameter should be an integer"

        self.obj = obj
        self.n_series = None
        self.lags = lags
        self.type_pi = type_pi
        self.level = level
        if self.type_pi == "quantile":
            self.obj = QuantileRegressor(
                self.obj, level=self.level, scoring="conformal"
            )
        self.block_size = block_size
        self.replications = replications
        self.kernel = kernel
        self.agg = agg
        self.verbose = verbose
        self.show_progress = show_progress
        self.series_names = ["series0"]
        self.input_dates = None
        self.quantiles = None
        self.fit_objs_ = {}
        self.y_ = None  # MTS responses (most recent observations first)
        self.X_ = None  # MTS lags
        self.xreg_ = None
        self.y_means_ = {}
        self.mean_ = None
        self.median_ = None
        self.upper_ = None
        self.lower_ = None
        self.output_dates_ = None
        self.preds_std_ = []
        self.gaussian_preds_std_ = None
        self.alpha_ = None
        self.return_std_ = None
        self.df_ = None
        self.residuals_ = []
        self.abs_calib_residuals_ = None
        self.calib_residuals_quantile_ = None
        self.residuals_sims_ = None
        self.kde_ = None
        self.sims_ = None
        self.residuals_std_dev_ = None
        self.n_obs_ = None
        self.level_ = None
        self.init_n_series_ = None

    def fit(self, X, xreg=None, **kwargs):
        """Fit MTS model to training data X, with optional regressors xreg

        Parameters:

        X: {array-like}, shape = [n_samples, n_features]
            Training time series, where n_samples is the number
            of samples and n_features is the number of features;
            X must be in increasing order (most recent observations last)

        xreg: {array-like}, shape = [n_samples, n_features_xreg]
            Additional (external) regressors to be passed to self.obj
            xreg must be in 'increasing' order (most recent observations last)

        **kwargs: for now, additional parameters to be passed to for kernel density estimation, when needed (see sklearn.neighbors.KernelDensity)

        Returns:

        self: object
        """
        try:
            self.init_n_series_ = X.shape[1]
        except IndexError as e:
            self.init_n_series_ = 1

        # Automatic lag selection if requested
        if isinstance(self.lags, str):
            max_lags = min(25, X.shape[0] // 4)
            best_ic = float("inf")
            best_lags = 1

            if self.verbose:
                print(
                    f"\nSelecting optimal number of lags using {self.lags}..."
                )
                iterator = tqdm(range(1, max_lags + 1))
            else:
                iterator = range(1, max_lags + 1)

            for lag in iterator:
                # Convert DataFrame to numpy array before reversing
                if isinstance(X, pd.DataFrame):
                    X_values = X.values[::-1]
                else:
                    X_values = X[::-1]

                # Try current lag value
                if self.init_n_series_ > 1:
                    mts_input = ts.create_train_inputs(X_values, lag)
                else:
                    mts_input = ts.create_train_inputs(
                        X_values.reshape(-1, 1), lag
                    )

                # Cook training set and fit model
                dummy_y, scaled_Z = self.cook_training_set(
                    y=np.ones(mts_input[0].shape[0]), X=mts_input[1]
                )
                residuals_ = []

                for i in range(self.init_n_series_):
                    y_mean = np.mean(mts_input[0][:, i])
                    centered_y_i = mts_input[0][:, i] - y_mean
                    self.obj.fit(X=scaled_Z, y=centered_y_i)
                    residuals_.append(
                        (centered_y_i - self.obj.predict(scaled_Z)).tolist()
                    )

                self.residuals_ = np.asarray(residuals_).T
                ic = self._compute_information_criterion(
                    curr_lags=lag, criterion=self.lags
                )

                if self.verbose:
                    print(f"Trying lags={lag}, {self.lags}={ic:.2f}")

                if ic < best_ic:
                    best_ic = ic
                    best_lags = lag

            if self.verbose:
                print(
                    f"\nSelected {best_lags} lags with {self.lags}={best_ic:.2f}"
                )

            self.lags = best_lags

        self.input_dates = None
        self.df_ = None

        if isinstance(X, pd.DataFrame) is False:
            # input data set is a numpy array
            if xreg is None:
                X = pd.DataFrame(X)
                self.series_names = [
                    "series" + str(i) for i in range(X.shape[1])
                ]
            else:
                # xreg is not None
                X = mo.cbind(X, xreg)
                self.xreg_ = xreg

        else:  # input data set is a DataFrame with column names
            X_index = None
            if X.index is not None:
                X_index = X.index
            if xreg is None:
                X = copy.deepcopy(mo.convert_df_to_numeric(X))
            else:
                X = copy.deepcopy(mo.cbind(mo.convert_df_to_numeric(X), xreg))
                self.xreg_ = xreg
            if X_index is not None:
                X.index = X_index
            self.series_names = X.columns.tolist()

        if isinstance(X, pd.DataFrame):
            if self.df_ is None:
                self.df_ = X
                X = X.values
            else:
                input_dates_prev = pd.DatetimeIndex(self.df_.index.values)
                frequency = pd.infer_freq(input_dates_prev)
                self.df_ = pd.concat([self.df_, X], axis=0)
                self.input_dates = pd.date_range(
                    start=input_dates_prev[0],
                    periods=len(input_dates_prev) + X.shape[0],
                    freq=frequency,
                ).values.tolist()
                self.df_.index = self.input_dates
                X = self.df_.values
            self.df_.columns = self.series_names
        else:
            if self.df_ is None:
                self.df_ = pd.DataFrame(X, columns=self.series_names)
            else:
                self.df_ = pd.concat(
                    [self.df_, pd.DataFrame(X, columns=self.series_names)],
                    axis=0,
                )

        self.input_dates = ts.compute_input_dates(self.df_)

        try:
            # multivariate time series
            n, p = X.shape
        except:
            # univariate time series
            n = X.shape[0]
            p = 1
        self.n_obs_ = n

        rep_1_n = np.repeat(1, n)

        self.y_ = None
        self.X_ = None
        self.n_series = p
        self.fit_objs_.clear()
        self.y_means_.clear()
        residuals_ = []
        self.residuals_ = None
        self.residuals_sims_ = None
        self.kde_ = None
        self.sims_ = None
        self.scaled_Z_ = None
        self.centered_y_is_ = []

        if self.init_n_series_ > 1:
            # multivariate time series
            mts_input = ts.create_train_inputs(X[::-1], self.lags)
        else:
            # univariate time series
            mts_input = ts.create_train_inputs(
                X.reshape(-1, 1)[::-1], self.lags
            )

        self.y_ = mts_input[0]

        self.X_ = mts_input[1]

        dummy_y, scaled_Z = self.cook_training_set(y=rep_1_n, X=self.X_)

        self.scaled_Z_ = scaled_Z

        # loop on all the time series and adjust self.obj.fit
        if self.verbose > 0:
            print(
                f"\n Adjusting {type(self.obj).__name__} to multivariate time series... \n"
            )

        if self.show_progress is True:
            iterator = tqdm(range(self.init_n_series_))
        else:
            iterator = range(self.init_n_series_)

        if self.type_pi in (
            "gaussian",
            "kde",
            "bootstrap",
            "block-bootstrap",
        ) or self.type_pi.startswith("vine"):
            for i in iterator:
                y_mean = np.mean(self.y_[:, i])
                self.y_means_[i] = y_mean
                centered_y_i = self.y_[:, i] - y_mean
                self.centered_y_is_.append(centered_y_i)
                self.obj.fit(X=scaled_Z, y=centered_y_i)
                self.fit_objs_[i] = deepcopy(self.obj)
                residuals_.append(
                    (
                        centered_y_i - self.fit_objs_[i].predict(scaled_Z)
                    ).tolist()
                )

        if self.type_pi == "quantile":
            for i in iterator:
                y_mean = np.mean(self.y_[:, i])
                self.y_means_[i] = y_mean
                centered_y_i = self.y_[:, i] - y_mean
                self.centered_y_is_.append(centered_y_i)
                self.obj.fit(X=scaled_Z, y=centered_y_i)
                self.fit_objs_[i] = deepcopy(self.obj)

        if self.type_pi.startswith("scp"):
            # split conformal prediction
            for i in iterator:
                n_y = self.y_.shape[0]
                n_y_half = n_y // 2
                first_half_idx = range(0, n_y_half)
                second_half_idx = range(n_y_half, n_y)
                y_mean_temp = np.mean(self.y_[first_half_idx, i])
                centered_y_i_temp = self.y_[first_half_idx, i] - y_mean_temp
                self.obj.fit(X=scaled_Z[first_half_idx, :], y=centered_y_i_temp)
                # calibrated residuals actually
                residuals_.append(
                    (
                        self.y_[second_half_idx, i]
                        - (
                            y_mean_temp
                            + self.obj.predict(scaled_Z[second_half_idx, :])
                        )
                    ).tolist()
                )
                # fit on the second half
                y_mean = np.mean(self.y_[second_half_idx, i])
                self.y_means_[i] = y_mean
                centered_y_i = self.y_[second_half_idx, i] - y_mean
                self.obj.fit(X=scaled_Z[second_half_idx, :], y=centered_y_i)
                self.fit_objs_[i] = deepcopy(self.obj)

        self.residuals_ = np.asarray(residuals_).T

        if self.type_pi == "gaussian":
            self.gaussian_preds_std_ = np.std(self.residuals_, axis=0)

        if self.type_pi.startswith("scp2"):
            # Calculate mean and standard deviation for each column
            data_mean = np.mean(self.residuals_, axis=0)
            self.residuals_std_dev_ = np.std(self.residuals_, axis=0)
            # Center and scale the array using broadcasting
            self.residuals_ = (
                self.residuals_ - data_mean[np.newaxis, :]
            ) / self.residuals_std_dev_[np.newaxis, :]

        if self.replications != None and "kde" in self.type_pi:
            if self.verbose > 0:
                print(f"\n Simulate residuals using {self.kernel} kernel... \n")
            assert self.kernel in (
                "gaussian",
                "tophat",
            ), "currently, 'kernel' must be either 'gaussian' or 'tophat'"
            kernel_bandwidths = {"bandwidth": np.logspace(-6, 6, 150)}
            grid = GridSearchCV(
                KernelDensity(kernel=self.kernel, **kwargs),
                param_grid=kernel_bandwidths,
            )
            grid.fit(self.residuals_)

            if self.verbose > 0:
                print(
                    f"\n Best parameters for {self.kernel} kernel: {grid.best_params_} \n"
                )

            self.kde_ = grid.best_estimator_

        return self

    def partial_fit(self, X, xreg=None, **kwargs):
        """partial_fit MTS model to training data X, with optional regressors xreg

        Parameters:

        X: {array-like}, shape = [n_samples, n_features]
            Training time series, where n_samples is the number
            of samples and n_features is the number of features;
            X must be in increasing order (most recent observations last)

        xreg: {array-like}, shape = [n_samples, n_features_xreg]
            Additional (external) regressors to be passed to self.obj
            xreg must be in 'increasing' order (most recent observations last)

        **kwargs: for now, additional parameters to be passed to for kernel density estimation, when needed (see sklearn.neighbors.KernelDensity)

        Returns:

        self: object
        """
        try:
            self.init_n_series_ = X.shape[1]
        except IndexError as e:
            self.init_n_series_ = 1

        # Automatic lag selection if requested
        if isinstance(self.lags, str):
            max_lags = min(25, X.shape[0] // 4)
            best_ic = float("inf")
            best_lags = 1

            if self.verbose:
                print(
                    f"\nSelecting optimal number of lags using {self.lags}..."
                )
                iterator = tqdm(range(1, max_lags + 1))
            else:
                iterator = range(1, max_lags + 1)

            for lag in iterator:
                # Convert DataFrame to numpy array before reversing
                if isinstance(X, pd.DataFrame):
                    X_values = X.values[::-1]
                else:
                    X_values = X[::-1]

                # Try current lag value
                if self.init_n_series_ > 1:
                    mts_input = ts.create_train_inputs(X_values, lag)
                else:
                    mts_input = ts.create_train_inputs(
                        X_values.reshape(-1, 1), lag
                    )

                # Cook training set and partial_fit model
                dummy_y, scaled_Z = self.cook_training_set(
                    y=np.ones(mts_input[0].shape[0]), X=mts_input[1]
                )
                residuals_ = []

                for i in range(self.init_n_series_):
                    y_mean = np.mean(mts_input[0][:, i])
                    centered_y_i = mts_input[0][:, i] - y_mean
                    self.obj.partial_fit(X=scaled_Z, y=centered_y_i)
                    residuals_.append(
                        (centered_y_i - self.obj.predict(scaled_Z)).tolist()
                    )

                self.residuals_ = np.asarray(residuals_).T
                ic = self._compute_information_criterion(
                    curr_lags=lag, criterion=self.lags
                )

                if self.verbose:
                    print(f"Trying lags={lag}, {self.lags}={ic:.2f}")

                if ic < best_ic:
                    best_ic = ic
                    best_lags = lag

            if self.verbose:
                print(
                    f"\nSelected {best_lags} lags with {self.lags}={best_ic:.2f}"
                )

            self.lags = best_lags

        self.input_dates = None
        self.df_ = None

        if isinstance(X, pd.DataFrame) is False:
            # input data set is a numpy array
            if xreg is None:
                X = pd.DataFrame(X)
                if len(X.shape) > 1:
                    self.series_names = [
                        "series" + str(i) for i in range(X.shape[1])
                    ]
                else:
                    self.series_names = ["series0"]
            else:
                # xreg is not None
                X = mo.cbind(X, xreg)
                self.xreg_ = xreg

        else:  # input data set is a DataFrame with column names
            X_index = None
            if X.index is not None:
                X_index = X.index
            if xreg is None:
                X = copy.deepcopy(mo.convert_df_to_numeric(X))
            else:
                X = copy.deepcopy(mo.cbind(mo.convert_df_to_numeric(X), xreg))
                self.xreg_ = xreg
            if X_index is not None:
                X.index = X_index
            self.series_names = X.columns.tolist()

        if isinstance(X, pd.DataFrame):
            if self.df_ is None:
                self.df_ = X
                X = X.values
            else:
                input_dates_prev = pd.DatetimeIndex(self.df_.index.values)
                frequency = pd.infer_freq(input_dates_prev)
                self.df_ = pd.concat([self.df_, X], axis=0)
                self.input_dates = pd.date_range(
                    start=input_dates_prev[0],
                    periods=len(input_dates_prev) + X.shape[0],
                    freq=frequency,
                ).values.tolist()
                self.df_.index = self.input_dates
                X = self.df_.values
            self.df_.columns = self.series_names
        else:
            if self.df_ is None:
                self.df_ = pd.DataFrame(X, columns=self.series_names)
            else:
                self.df_ = pd.concat(
                    [self.df_, pd.DataFrame(X, columns=self.series_names)],
                    axis=0,
                )

        self.input_dates = ts.compute_input_dates(self.df_)

        try:
            # multivariate time series
            n, p = X.shape
        except:
            # univariate time series
            n = X.shape[0]
            p = 1
        self.n_obs_ = n

        rep_1_n = np.repeat(1, n)

        self.y_ = None
        self.X_ = None
        self.n_series = p
        self.fit_objs_.clear()
        self.y_means_.clear()
        residuals_ = []
        self.residuals_ = None
        self.residuals_sims_ = None
        self.kde_ = None
        self.sims_ = None
        self.scaled_Z_ = None
        self.centered_y_is_ = []

        if self.init_n_series_ > 1:
            # multivariate time series
            mts_input = ts.create_train_inputs(X[::-1], self.lags)
        else:
            # univariate time series
            mts_input = ts.create_train_inputs(
                X.reshape(-1, 1)[::-1], self.lags
            )

        self.y_ = mts_input[0]

        self.X_ = mts_input[1]

        dummy_y, scaled_Z = self.cook_training_set(y=rep_1_n, X=self.X_)

        self.scaled_Z_ = scaled_Z

        # loop on all the time series and adjust self.obj.partial_fit
        if self.verbose > 0:
            print(
                f"\n Adjusting {type(self.obj).__name__} to multivariate time series... \n"
            )

        if self.show_progress is True:
            iterator = tqdm(range(self.init_n_series_))
        else:
            iterator = range(self.init_n_series_)

        if self.type_pi in (
            "gaussian",
            "kde",
            "bootstrap",
            "block-bootstrap",
        ) or self.type_pi.startswith("vine"):
            for i in iterator:
                y_mean = np.mean(self.y_[:, i])
                self.y_means_[i] = y_mean
                centered_y_i = self.y_[:, i] - y_mean
                self.centered_y_is_.append(centered_y_i)
                self.obj.partial_fit(X=scaled_Z, y=centered_y_i)
                self.fit_objs_[i] = deepcopy(self.obj)
                residuals_.append(
                    (
                        centered_y_i - self.fit_objs_[i].predict(scaled_Z)
                    ).tolist()
                )

        if self.type_pi == "quantile":
            for i in iterator:
                y_mean = np.mean(self.y_[:, i])
                self.y_means_[i] = y_mean
                centered_y_i = self.y_[:, i] - y_mean
                self.centered_y_is_.append(centered_y_i)
                self.obj.partial_fit(X=scaled_Z, y=centered_y_i)
                self.fit_objs_[i] = deepcopy(self.obj)

        if self.type_pi.startswith("scp"):
            # split conformal prediction
            for i in iterator:
                n_y = self.y_.shape[0]
                n_y_half = n_y // 2
                first_half_idx = range(0, n_y_half)
                second_half_idx = range(n_y_half, n_y)
                y_mean_temp = np.mean(self.y_[first_half_idx, i])
                centered_y_i_temp = self.y_[first_half_idx, i] - y_mean_temp
                self.obj.partial_fit(
                    X=scaled_Z[first_half_idx, :], y=centered_y_i_temp
                )
                # calibrated residuals actually
                residuals_.append(
                    (
                        self.y_[second_half_idx, i]
                        - (
                            y_mean_temp
                            + self.obj.predict(scaled_Z[second_half_idx, :])
                        )
                    ).tolist()
                )
                # partial_fit on the second half
                y_mean = np.mean(self.y_[second_half_idx, i])
                self.y_means_[i] = y_mean
                centered_y_i = self.y_[second_half_idx, i] - y_mean
                self.obj.partial_fit(
                    X=scaled_Z[second_half_idx, :], y=centered_y_i
                )
                self.fit_objs_[i] = deepcopy(self.obj)

        self.residuals_ = np.asarray(residuals_).T

        if self.type_pi == "gaussian":
            self.gaussian_preds_std_ = np.std(self.residuals_, axis=0)

        if self.type_pi.startswith("scp2"):
            # Calculate mean and standard deviation for each column
            data_mean = np.mean(self.residuals_, axis=0)
            self.residuals_std_dev_ = np.std(self.residuals_, axis=0)
            # Center and scale the array using broadcasting
            self.residuals_ = (
                self.residuals_ - data_mean[np.newaxis, :]
            ) / self.residuals_std_dev_[np.newaxis, :]

        if self.replications != None and "kde" in self.type_pi:
            if self.verbose > 0:
                print(f"\n Simulate residuals using {self.kernel} kernel... \n")
            assert self.kernel in (
                "gaussian",
                "tophat",
            ), "currently, 'kernel' must be either 'gaussian' or 'tophat'"
            kernel_bandwidths = {"bandwidth": np.logspace(-6, 6, 150)}
            grid = GridSearchCV(
                KernelDensity(kernel=self.kernel, **kwargs),
                param_grid=kernel_bandwidths,
            )
            grid.fit(self.residuals_)

            if self.verbose > 0:
                print(
                    f"\n Best parameters for {self.kernel} kernel: {grid.best_params_} \n"
                )

            self.kde_ = grid.best_estimator_

        return self

    def _predict_quantiles(self, h, quantiles, **kwargs):
        """Predict arbitrary quantiles from simulated paths."""
        # Ensure output dates are set
        self.output_dates_, _ = ts.compute_output_dates(self.df_, h)

        # Trigger full prediction to generate self.sims_
        if not hasattr(self, "sims_") or self.sims_ is None:
            _ = self.predict(h=h, level=95, **kwargs)  # Any level triggers sim

        result_dict = {}

        # Stack simulations: (R, h, n_series)
        sims_array = np.stack([sim.values for sim in self.sims_], axis=0)

        # Compute quantiles over replication axis
        q_values = np.quantile(
            sims_array, quantiles, axis=0
        )  # (n_q, h, n_series)

        for i, q in enumerate(quantiles):
            # Clean label: 0.05 → "05", 0.1 → "10", 0.95 → "95"
            q_label = (
                f"{int(q * 100):02d}"
                if (q * 100).is_integer()
                else f"{q:.3f}".replace(".", "_")
            )
            for series_id in range(self.init_n_series_):
                series_name = self.series_names[series_id]
                col_name = f"quantile_{q_label}_{series_name}"
                result_dict[col_name] = q_values[i, :, series_id]

        df_return_quantiles = pd.DataFrame(
            result_dict, index=self.output_dates_
        )

        return df_return_quantiles

    def predict(self, h=5, level=95, quantiles=None, **kwargs):
        """Forecast all the time series, h steps ahead"""

        if quantiles is not None:
            # Validate
            quantiles = np.asarray(quantiles)
            if not ((quantiles > 0) & (quantiles < 1)).all():
                raise ValueError("quantiles must be between 0 and 1.")
            # Delegate to dedicated method
            return self._predict_quantiles(h=h, quantiles=quantiles, **kwargs)

        if isinstance(level, list) or isinstance(level, np.ndarray):
            # Store results
            result_dict = {}
            # Loop through alphas and calculate lower/upper for each alpha level
            # E.g [0.5, 2.5, 5, 16.5, 25, 50]
            for lev in level:
                # Get the forecast for this alpha
                res = self.predict(h=h, level=lev, **kwargs)
                # Adjust index and collect lower/upper bounds
                res.lower.index = pd.to_datetime(res.lower.index)
                res.upper.index = pd.to_datetime(res.upper.index)
                # Loop over each time series (multivariate) and flatten results
                if isinstance(res.lower, pd.DataFrame):
                    for (
                        series
                    ) in (
                        res.lower.columns
                    ):  # Assumes 'lower' and 'upper' have multiple series
                        result_dict[f"lower_{lev}_{series}"] = (
                            res.lower[series].to_numpy().flatten()
                        )
                        result_dict[f"upper_{lev}_{series}"] = (
                            res.upper[series].to_numpy().flatten()
                        )
                else:
                    for series_id in range(
                        self.n_series
                    ):  # Assumes 'lower' and 'upper' have multiple series
                        result_dict[f"lower_{lev}_{series_id}"] = (
                            res.lower[series_id, :].to_numpy().flatten()
                        )
                        result_dict[f"upper_{lev}_{series_id}"] = (
                            res.upper[series_id, :].to_numpy().flatten()
                        )
            return pd.DataFrame(result_dict, index=self.output_dates_)

        # only one prediction interval
        self.output_dates_, frequency = ts.compute_output_dates(self.df_, h)

        self.level_ = level

        self.return_std_ = False  # do not remove (/!\)

        self.mean_ = None  # do not remove (/!\)

        self.mean_ = deepcopy(self.y_)  # do not remove (/!\)

        self.lower_ = None  # do not remove (/!\)

        self.upper_ = None  # do not remove (/!\)

        self.sims_ = None  # do not remove (/!\)

        y_means_ = np.asarray(
            [self.y_means_[i] for i in range(self.init_n_series_)]
        )

        n_features = self.init_n_series_ * self.lags

        self.alpha_ = 100 - level

        pi_multiplier = norm.ppf(1 - self.alpha_ / 200)

        if "return_std" in kwargs:  # bayesian forecasting
            self.return_std_ = True
            self.preds_std_ = []
            DescribeResult = namedtuple(
                "DescribeResult", ("mean", "lower", "upper")
            )  # to be updated

        if "return_pi" in kwargs:  # split conformal, without simulation
            mean_pi_ = []
            lower_pi_ = []
            upper_pi_ = []
            median_pi_ = []
            DescribeResult = namedtuple(
                "DescribeResult", ("mean", "lower", "upper")
            )  # to be updated

        if self.kde_ != None and "kde" in self.type_pi:  # kde
            target_cols = self.df_.columns[
                : self.init_n_series_
            ]  # Get target column names
            if self.verbose == 1:
                self.residuals_sims_ = tuple(
                    self.kde_.sample(
                        n_samples=h, random_state=self.seed + 100 * i
                    )  # Keep full sample
                    for i in tqdm(range(self.replications))
                )
            elif self.verbose == 0:
                self.residuals_sims_ = tuple(
                    self.kde_.sample(
                        n_samples=h, random_state=self.seed + 100 * i
                    )  # Keep full sample
                    for i in range(self.replications)
                )

            # Convert to DataFrames after sampling
            self.residuals_sims_ = tuple(
                pd.DataFrame(
                    sim,  # Keep all columns
                    columns=target_cols,  # Use original target column names
                    index=self.output_dates_,
                )
                for sim in self.residuals_sims_
            )

        if self.type_pi in ("bootstrap", "scp-bootstrap", "scp2-bootstrap"):
            assert self.replications is not None and isinstance(
                self.replications, int
            ), "'replications' must be provided and be an integer"
            if self.verbose == 1:
                self.residuals_sims_ = tuple(
                    ts.bootstrap(
                        self.residuals_,
                        h=h,
                        block_size=None,
                        seed=self.seed + 100 * i,
                    )
                    for i in tqdm(range(self.replications))
                )
            elif self.verbose == 0:
                self.residuals_sims_ = tuple(
                    ts.bootstrap(
                        self.residuals_,
                        h=h,
                        block_size=None,
                        seed=self.seed + 100 * i,
                    )
                    for i in range(self.replications)
                )

        if self.type_pi in (
            "block-bootstrap",
            "scp-block-bootstrap",
            "scp2-block-bootstrap",
        ):
            if self.block_size is None:
                self.block_size = int(
                    np.ceil(3.15 * (self.residuals_.shape[0] ** (1 / 3)))
                )

            assert self.replications is not None and isinstance(
                self.replications, int
            ), "'replications' must be provided and be an integer"
            if self.verbose == 1:
                self.residuals_sims_ = tuple(
                    ts.bootstrap(
                        self.residuals_,
                        h=h,
                        block_size=self.block_size,
                        seed=self.seed + 100 * i,
                    )
                    for i in tqdm(range(self.replications))
                )
            elif self.verbose == 0:
                self.residuals_sims_ = tuple(
                    ts.bootstrap(
                        self.residuals_,
                        h=h,
                        block_size=self.block_size,
                        seed=self.seed + 100 * i,
                    )
                    for i in range(self.replications)
                )

        if "vine" in self.type_pi:
            if self.verbose == 1:
                self.residuals_sims_ = tuple(
                    vinecopula_sample(
                        x=self.residuals_,
                        n_samples=h,
                        method=self.type_pi,
                        random_state=self.seed + 100 * i,
                    )
                    for i in tqdm(range(self.replications))
                )
            elif self.verbose == 0:
                self.residuals_sims_ = tuple(
                    vinecopula_sample(
                        x=self.residuals_,
                        n_samples=h,
                        method=self.type_pi,
                        random_state=self.seed + 100 * i,
                    )
                    for i in range(self.replications)
                )

        mean_ = deepcopy(self.mean_)

        for i in range(h):
            new_obs = ts.reformat_response(mean_, self.lags)
            new_X = new_obs.reshape(1, -1)
            cooked_new_X = self.cook_test_set(new_X, **kwargs)

            if "return_std" in kwargs:
                self.preds_std_.append(
                    [
                        np.asarray(
                            self.fit_objs_[i].predict(
                                cooked_new_X, return_std=True
                            )[1]
                        ).item()
                        for i in range(self.n_series)
                    ]
                )

            if "return_pi" in kwargs:
                for i in range(self.n_series):
                    preds_pi = self.fit_objs_[i].predict(cooked_new_X, **kwargs)
                    mean_pi_.append(preds_pi.mean[0])
                    lower_pi_.append(preds_pi.lower[0])
                    upper_pi_.append(preds_pi.upper[0])

            if self.type_pi != "quantile":
                predicted_cooked_new_X = np.asarray(
                    [
                        np.asarray(
                            self.fit_objs_[i].predict(cooked_new_X)
                        ).item()
                        for i in range(self.init_n_series_)
                    ]
                )
            else:
                predicted_cooked_new_X = np.asarray(
                    [
                        np.asarray(
                            self.fit_objs_[i]
                            .predict(cooked_new_X, return_pi=True)
                            .upper
                        ).item()
                        for i in range(self.init_n_series_)
                    ]
                )

            preds = np.asarray(y_means_ + predicted_cooked_new_X)

            # Create full row with both predictions and external regressors
            if self.xreg_ is not None and "xreg" in kwargs:
                next_xreg = kwargs["xreg"].iloc[i: i + 1].values.flatten()
                full_row = np.concatenate([preds, next_xreg])
            else:
                full_row = preds

            # Create a new row with same number of columns as mean_
            new_row = np.zeros((1, mean_.shape[1]))
            new_row[0, : full_row.shape[0]] = full_row

            # Maintain the full dimensionality by using vstack instead of rbind
            mean_ = np.vstack([new_row, mean_[:-1]])

        # Final output should only include the target columns
        self.mean_ = pd.DataFrame(
            mean_[0: min(h, self.n_obs_ - self.lags), : self.init_n_series_][
                ::-1
            ],
            columns=self.df_.columns[: self.init_n_series_],
            index=self.output_dates_,
        )

        # function's return ----------------------------------------------------------------------
        if (
            (("return_std" not in kwargs) and ("return_pi" not in kwargs))
            and (self.type_pi not in ("gaussian", "scp"))
        ) or ("vine" in self.type_pi):
            if self.replications is None:
                return self.mean_.iloc[:, : self.init_n_series_]

            # if "return_std" not in kwargs and self.replications is not None
            meanf = []
            medianf = []
            lower = []
            upper = []

            if "scp2" in self.type_pi:
                if self.verbose == 1:
                    self.sims_ = tuple(
                        (
                            self.mean_
                            + self.residuals_sims_[i]
                            * self.residuals_std_dev_[np.newaxis, :]
                            for i in tqdm(range(self.replications))
                        )
                    )
                elif self.verbose == 0:
                    self.sims_ = tuple(
                        (
                            self.mean_
                            + self.residuals_sims_[i]
                            * self.residuals_std_dev_[np.newaxis, :]
                            for i in range(self.replications)
                        )
                    )
            else:
                if self.verbose == 1:
                    self.sims_ = tuple(
                        (
                            self.mean_ + self.residuals_sims_[i]
                            for i in tqdm(range(self.replications))
                        )
                    )
                elif self.verbose == 0:
                    self.sims_ = tuple(
                        (
                            self.mean_ + self.residuals_sims_[i]
                            for i in range(self.replications)
                        )
                    )

            DescribeResult = namedtuple(
                "DescribeResult", ("mean", "sims", "lower", "upper")
            )
            for ix in range(self.init_n_series_):
                sims_ix = getsims(self.sims_, ix)
                if self.agg == "mean":
                    meanf.append(np.mean(sims_ix, axis=1))
                else:
                    medianf.append(np.median(sims_ix, axis=1))
                lower.append(np.quantile(sims_ix, q=self.alpha_ / 200, axis=1))
                upper.append(
                    np.quantile(sims_ix, q=1 - self.alpha_ / 200, axis=1)
                )
            self.mean_ = pd.DataFrame(
                np.asarray(meanf).T,
                columns=self.series_names[
                    : self.init_n_series_
                ],  # self.df_.columns,
                index=self.output_dates_,
            )

            self.lower_ = pd.DataFrame(
                np.asarray(lower).T,
                columns=self.series_names[
                    : self.init_n_series_
                ],  # self.df_.columns,
                index=self.output_dates_,
            )

            self.upper_ = pd.DataFrame(
                np.asarray(upper).T,
                columns=self.series_names[
                    : self.init_n_series_
                ],  # self.df_.columns,
                index=self.output_dates_,
            )

            try:
                self.median_ = pd.DataFrame(
                    np.asarray(medianf).T,
                    columns=self.series_names[
                        : self.init_n_series_
                    ],  # self.df_.columns,
                    index=self.output_dates_,
                )
            except Exception as e:
                pass

            return DescribeResult(
                self.mean_, self.sims_, self.lower_, self.upper_
            )

        if (
            (("return_std" in kwargs) or ("return_pi" in kwargs))
            and (self.type_pi not in ("gaussian", "scp"))
        ) or "vine" in self.type_pi:
            DescribeResult = namedtuple(
                "DescribeResult", ("mean", "lower", "upper")
            )

            self.mean_ = pd.DataFrame(
                np.asarray(self.mean_),
                columns=self.series_names,  # self.df_.columns,
                index=self.output_dates_,
            )

            if "return_std" in kwargs:
                self.preds_std_ = np.asarray(self.preds_std_)

                self.lower_ = pd.DataFrame(
                    self.mean_.values - pi_multiplier * self.preds_std_,
                    columns=self.series_names,  # self.df_.columns,
                    index=self.output_dates_,
                )

                self.upper_ = pd.DataFrame(
                    self.mean_.values + pi_multiplier * self.preds_std_,
                    columns=self.series_names,  # self.df_.columns,
                    index=self.output_dates_,
                )

            if "return_pi" in kwargs:
                self.lower_ = pd.DataFrame(
                    np.asarray(lower_pi_).reshape(h, self.n_series)
                    + y_means_[np.newaxis, :],
                    columns=self.series_names,  # self.df_.columns,
                    index=self.output_dates_,
                )

                self.upper_ = pd.DataFrame(
                    np.asarray(upper_pi_).reshape(h, self.n_series)
                    + y_means_[np.newaxis, :],
                    columns=self.series_names,  # self.df_.columns,
                    index=self.output_dates_,
                )

            res = DescribeResult(self.mean_, self.lower_, self.upper_)

            if self.xreg_ is not None:
                if len(self.xreg_.shape) > 1:
                    res2 = mx.tuple_map(
                        res,
                        lambda x: mo.delete_last_columns(
                            x, num_columns=self.xreg_.shape[1]
                        ),
                    )
                else:
                    res2 = mx.tuple_map(
                        res, lambda x: mo.delete_last_columns(x, num_columns=1)
                    )
                return DescribeResult(res2[0], res2[1], res2[2])

            return res

        if self.type_pi == "gaussian":
            DescribeResult = namedtuple(
                "DescribeResult", ("mean", "lower", "upper")
            )

            self.mean_ = pd.DataFrame(
                np.asarray(self.mean_),
                columns=self.series_names,  # self.df_.columns,
                index=self.output_dates_,
            )

            # Use Bayesian std if available, otherwise use gaussian residual std
            if "return_std" in kwargs and len(self.preds_std_) > 0:
                preds_std_to_use = np.asarray(self.preds_std_)
            else:
                preds_std_to_use = self.gaussian_preds_std_

            self.lower_ = pd.DataFrame(
                self.mean_.values - pi_multiplier * preds_std_to_use,
                columns=self.series_names,  # self.df_.columns,
                index=self.output_dates_,
            )

            self.upper_ = pd.DataFrame(
                self.mean_.values + pi_multiplier * preds_std_to_use,
                columns=self.series_names,  # self.df_.columns,
                index=self.output_dates_,
            )

            res = DescribeResult(self.mean_, self.lower_, self.upper_)

            if self.xreg_ is not None:
                if len(self.xreg_.shape) > 1:
                    res2 = mx.tuple_map(
                        res,
                        lambda x: mo.delete_last_columns(
                            x, num_columns=self.xreg_.shape[1]
                        ),
                    )
                else:
                    res2 = mx.tuple_map(
                        res, lambda x: mo.delete_last_columns(x, num_columns=1)
                    )
                return DescribeResult(res2[0], res2[1], res2[2])

            return res

        if self.type_pi == "quantile":
            DescribeResult = namedtuple("DescribeResult", ("mean"))

            self.mean_ = pd.DataFrame(
                np.asarray(self.mean_),
                columns=self.series_names,  # self.df_.columns,
                index=self.output_dates_,
            )

            res = DescribeResult(self.mean_)

            if self.xreg_ is not None:
                if len(self.xreg_.shape) > 1:
                    res2 = mx.tuple_map(
                        res,
                        lambda x: mo.delete_last_columns(
                            x, num_columns=self.xreg_.shape[1]
                        ),
                    )
                else:
                    res2 = mx.tuple_map(
                        res, lambda x: mo.delete_last_columns(x, num_columns=1)
                    )
                return DescribeResult(res2[0])

            return res

        # After prediction loop, ensure sims only contain target columns
        if self.sims_ is not None:
            if self.verbose == 1:
                self.sims_ = tuple(
                    sim[:h,]  # Only keep target columns and h rows
                    for sim in tqdm(self.sims_)
                )
            elif self.verbose == 0:
                self.sims_ = tuple(
                    sim[:h,]  # Only keep target columns and h rows
                    for sim in self.sims_
                )

            # Convert numpy arrays to DataFrames with proper columns
            self.sims_ = tuple(
                pd.DataFrame(
                    sim,
                    columns=self.df_.columns[: self.init_n_series_],
                    index=self.output_dates_,
                )
                for sim in self.sims_
            )

        if self.type_pi in (
            "kde",
            "bootstrap",
            "block-bootstrap",
            "vine-copula",
        ):
            if self.xreg_ is not None:
                # Use getsimsxreg when external regressors are present
                target_cols = self.df_.columns[: self.init_n_series_]
                self.sims_ = getsimsxreg(
                    self.sims_, self.output_dates_, target_cols
                )
            else:
                # Use original getsims for backward compatibility
                self.sims_ = getsims(self.sims_)

    def _crps_ensemble(self, y_true, simulations, axis=0):
        """
        Compute the Continuous Ranked Probability Score (CRPS) for an ensemble of simulations.

        The CRPS is a measure of the distance between the cumulative distribution
        function (CDF) of a forecast and the CDF of the observed value. This method
        computes the CRPS in a vectorized form for an ensemble of simulations, efficiently
        handling the case where there is only one simulation.

        Parameters
        ----------
        y_true : array_like, shape (n,)
            A 1D array of true values (observations).
            Each element represents the true value for a given sample.

        simulations : array_like, shape (n, R)
            A 2D array of simulated values. Each row corresponds to a different sample
            and each column corresponds to a different simulation of that sample.

        axis : int, optional, default=0
            Axis along which to transpose the simulations if needed.
            If axis=0, the simulations are transposed to shape (R, n).

        Returns
        -------
        crps : ndarray, shape (n,)
            A 1D array of CRPS scores, one for each sample.

        Notes
        -----
        The CRPS score is computed as:

        CRPS(y_true, simulations) = E[|X - y|] - 0.5 * E[|X - X'|]

        Where:
        - `X` is the ensemble of simulations.
        - `y` is the true value.
        - `X'` is a second independent sample from the ensemble.

        The calculation is vectorized to optimize performance for large datasets.

        The edge case where `R=1` (only one simulation) is handled by returning
        only `term1` (i.e., no ensemble spread).
        """
        sims = np.asarray(simulations)  # Convert simulations to numpy array
        if axis == 0:
            sims = sims.T  # Transpose if the axis is 0
        n, R = sims.shape  # n = number of samples, R = number of simulations
        # Term 1: E|X - y|, average absolute difference between simulations and true value
        term1 = np.mean(np.abs(sims - y_true[:, np.newaxis]), axis=1)
        # Handle edge case: if R == 1, return term1 (no spread in ensemble)
        if R == 1:
            return term1
        # Term 2: 0.5 * E|X - X'|, using efficient sorted formula
        sims_sorted = np.sort(sims, axis=1)  # Sort simulations along each row
        # Correct coefficients for efficient calculation
        j = np.arange(R)  # 0-indexed positions in the sorted simulations
        coefficients = (2 * (j + 1) - R - 1) / (
            R * (R - 1)
        )  # Efficient coefficient calculation
        # Dot product along the second axis (over the simulations)
        term2 = np.dot(sims_sorted, coefficients)
        # Return CRPS score: term1 - 0.5 * term2
        return term1 - 0.5 * term2

    def score(
        self,
        X,
        training_index,
        testing_index,
        scoring=None,
        alpha=0.5,
        **kwargs,
    ):
        """Train on training_index, score on testing_index."""

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

        if scoring == "pinball":
            # Predict requested quantile
            q_pred = self.predict(h=h, quantiles=[alpha], **kwargs)
            # Handle multivariate
            scores = []
            for j in range(p):
                series_name = getattr(self, "series_names", [f"Series_{j}"])[j]
                q_label = (
                    f"{int(alpha * 100):02d}"
                    if (alpha * 100).is_integer()
                    else f"{alpha:.3f}".replace(".", "_")
                )
                col = f"quantile_{q_label}_{series_name}"
                if col not in q_pred.columns:
                    raise ValueError(
                        f"Column '{col}' not found in quantile forecast output."
                    )
                y_true_j = X_test[:, j]
                y_pred_j = q_pred[col].values
                # Compute pinball loss for this series
                loss = mean_pinball_loss(y_true_j, y_pred_j, alpha=alpha)
                scores.append(loss)
            # Return average over series
            return np.mean(scores)

        if scoring == "crps":
            # Ensure simulations exist
            preds = self.predict(h=h, **kwargs)  # triggers self.sims_
            # Extract simulations: list of DataFrames → (R, h, p)
            sims_vals = np.stack(
                [sim.values for sim in self.sims_], axis=0
            )  # (R, h, p)
            crps_scores = []
            for j in range(p):
                y_true_j = X_test[:, j]
                sims_j = sims_vals[:, :, j]  # (R, h)
                crps_j = self._crps_ensemble(np.asarray(y_true_j), sims_j)
                crps_scores.append(np.mean(crps_j))  # average over horizon
            return np.mean(crps_scores)  # average over series

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
            "neg_mean_squared_error": lambda x, y: np.mean((x - y) ** 2),
            "neg_root_mean_squared_error": lambda x, y: np.sqrt(
                np.mean((x - y) ** 2)
            ),
            "neg_mean_squared_log_error": skm2.mean_squared_log_error,
            "neg_median_absolute_error": skm2.median_absolute_error,
            "r2": skm2.r2_score,
        }

        return scoring_options[scoring](X_test, preds)

    def plot(self, series=None, type_axis="dates", type_plot="pi"):
        """Plot time series forecast

        Parameters:

        series: {integer} or {string}
            series index or name

        """

        assert all(
            [
                self.mean_ is not None,
                self.lower_ is not None,
                self.upper_ is not None,
                self.output_dates_ is not None,
            ]
        ), "model forecasting must be obtained first (with predict)"

        if series is None:
            # assert (
            #    self.init_n_series_ == 1
            # ), "please specify series index or name (n_series > 1)"
            series = 0

        if isinstance(series, str):
            assert (
                series in self.series_names
            ), f"series {series} doesn't exist in the input dataset"
            series_idx = self.df_.columns.get_loc(series)
        else:
            assert isinstance(series, int) and (
                0 <= series < self.n_series
            ), f"check series index (< {self.n_series})"
            series_idx = series

        y_all = list(self.df_.iloc[:, series_idx]) + list(
            self.mean_.iloc[:, series_idx]
        )
        y_test = list(self.mean_.iloc[:, series_idx])
        n_points_all = len(y_all)
        n_points_train = self.df_.shape[0]

        if type_axis == "numeric":
            x_all = [i for i in range(n_points_all)]
            x_test = [i for i in range(n_points_train, n_points_all)]

        if type_axis == "dates":  # use dates
            x_all = np.concatenate(
                (self.input_dates.values, self.output_dates_.values), axis=None
            )
            x_test = self.output_dates_.values

        if type_plot == "pi":
            fig, ax = plt.subplots()
            ax.plot(x_all, y_all, "-")
            ax.plot(x_test, y_test, "-", color="orange")
            ax.fill_between(
                x_test,
                self.lower_.iloc[:, series_idx],
                self.upper_.iloc[:, series_idx],
                alpha=0.2,
                color="orange",
            )
            if self.replications is None:
                if self.n_series > 1:
                    plt.title(
                        f"prediction intervals for {series}",
                        loc="left",
                        fontsize=12,
                        fontweight=0,
                        color="black",
                    )
                else:
                    plt.title(
                        f"prediction intervals for input time series",
                        loc="left",
                        fontsize=12,
                        fontweight=0,
                        color="black",
                    )
                plt.show()
            else:  # self.replications is not None
                if self.n_series > 1:
                    plt.title(
                        f"prediction intervals for {self.replications} simulations of {series}",
                        loc="left",
                        fontsize=12,
                        fontweight=0,
                        color="black",
                    )
                else:
                    plt.title(
                        f"prediction intervals for {self.replications} simulations of input time series",
                        loc="left",
                        fontsize=12,
                        fontweight=0,
                        color="black",
                    )
                plt.show()

        if type_plot == "spaghetti":
            palette = plt.get_cmap("Set1")
            sims_ix = getsims(self.sims_, series_idx)
            plt.plot(x_all, y_all, "-")
            for col_ix in range(
                sims_ix.shape[1]
            ):  # avoid this when there are thousands of simulations
                plt.plot(
                    x_test,
                    sims_ix[:, col_ix],
                    "-",
                    color=palette(col_ix),
                    linewidth=1,
                    alpha=0.9,
                )
            plt.plot(x_all, y_all, "-", color="black")
            plt.plot(x_test, y_test, "-", color="blue")
            # Add titles
            if self.n_series > 1:
                plt.title(
                    f"{self.replications} simulations of {series}",
                    loc="left",
                    fontsize=12,
                    fontweight=0,
                    color="black",
                )
            else:
                plt.title(
                    f"{self.replications} simulations of input time series",
                    loc="left",
                    fontsize=12,
                    fontweight=0,
                    color="black",
                )
            plt.xlabel("Time")
            plt.ylabel("Values")
            # Show the graph
            plt.show()

    def cross_val_score(
        self,
        X,
        scoring="root_mean_squared_error",
        n_jobs=None,
        verbose=0,
        xreg=None,
        initial_window=5,
        horizon=3,
        fixed_window=False,
        show_progress=True,
        level=95,
        alpha=0.5,
        **kwargs,
    ):
        """Evaluate a score by time series cross-validation.

        Parameters:

            X: {array-like, sparse matrix} of shape (n_samples, n_features)
                The data to fit.

            scoring: str or a function
                A str in ('root_mean_squared_error', 'mean_squared_error', 'mean_error',
                'mean_absolute_error', 'mean_error', 'mean_percentage_error',
                'mean_absolute_percentage_error',  'winkler_score', 'coverage')
                Or a function defined as 'coverage' and 'winkler_score' in `utils.timeseries`

            n_jobs: int, default=None
                Number of jobs to run in parallel.

            verbose: int, default=0
                The verbosity level.

            xreg: array-like, optional (default=None)
                Additional (external) regressors to be passed to `fit`
                xreg must be in 'increasing' order (most recent observations last)

            initial_window: int
                initial number of consecutive values in each training set sample

            horizon: int
                number of consecutive values in test set sample

            fixed_window: boolean
                if False, all training samples start at index 0, and the training
                window's size is increasing.
                if True, the training window's size is fixed, and the window is
                rolling forward

            show_progress: boolean
                if True, a progress bar is printed

            level: int
                confidence level for prediction intervals

            alpha: float
                quantile level for pinball loss if scoring='pinball'
                0 < alpha < 1

            **kwargs: dict
                additional parameters to be passed to `fit` and `predict`

        Returns:

            A tuple: descriptive statistics or errors and raw errors

        """
        tscv = TimeSeriesSplit()

        tscv_obj = tscv.split(
            X,
            initial_window=initial_window,
            horizon=horizon,
            fixed_window=fixed_window,
        )

        if isinstance(scoring, str):
            assert scoring in (
                "pinball",
                "crps",
                "root_mean_squared_error",
                "mean_squared_error",
                "mean_error",
                "mean_absolute_error",
                "mean_percentage_error",
                "mean_absolute_percentage_error",
                "winkler_score",
                "coverage",
            ), "must have scoring in ('pinball', 'crps', 'root_mean_squared_error', 'mean_squared_error', 'mean_error', 'mean_absolute_error', 'mean_error', 'mean_percentage_error', 'mean_absolute_percentage_error',  'winkler_score', 'coverage')"

            def err_func(X_test, X_pred, scoring, alpha=0.5):
                if (self.replications is not None) or (
                    self.type_pi == "gaussian"
                ):  # probabilistic
                    if scoring == "pinball":
                        # Predict requested quantile
                        q_pred = self.predict(
                            h=len(X_test), quantiles=[alpha], **kwargs
                        )
                        # Handle multivariate
                        scores = []
                        p = X_test.shape[1] if len(X_test.shape) > 1 else 1
                        for j in range(p):
                            series_name = getattr(
                                self, "series_names", [f"Series_{j}"]
                            )[j]
                            q_label = (
                                f"{int(alpha * 100):02d}"
                                if (alpha * 100).is_integer()
                                else f"{alpha:.3f}".replace(".", "_")
                            )
                            col = f"quantile_{q_label}_{series_name}"
                            if col not in q_pred.columns:
                                raise ValueError(
                                    f"Column '{col}' not found in quantile forecast output."
                                )
                            try:
                                y_true_j = X_test[:, j] if p > 1 else X_test
                            except:
                                y_true_j = (
                                    X_test.iloc[:, j]
                                    if p > 1
                                    else X_test.values
                                )
                            y_pred_j = q_pred[col].values
                            # Compute pinball loss for this series
                            loss = mean_pinball_loss(
                                y_true_j, y_pred_j, alpha=alpha
                            )
                            scores.append(loss)
                        # Return average over series
                        return np.mean(scores)
                    elif scoring == "crps":
                        # Ensure simulations exist
                        _ = self.predict(
                            h=len(X_test), **kwargs
                        )  # triggers self.sims_
                        # Extract simulations: list of DataFrames → (R, h, p)
                        sims_vals = np.stack(
                            [sim.values for sim in self.sims_], axis=0
                        )  # (R, h, p)
                        crps_scores = []
                        p = X_test.shape[1] if len(X_test.shape) > 1 else 1
                        for j in range(p):
                            try:
                                y_true_j = X_test[:, j] if p > 1 else X_test
                            except Exception as e:
                                y_true_j = (
                                    X_test.iloc[:, j]
                                    if p > 1
                                    else X_test.values
                                )
                            sims_j = sims_vals[:, :, j]  # (R, h)
                            crps_j = self._crps_ensemble(
                                np.asarray(y_true_j), sims_j
                            )
                            crps_scores.append(
                                np.mean(crps_j)
                            )  # average over horizon
                        return np.mean(crps_scores)  # average over series
                    if scoring == "winkler_score":
                        return winkler_score(X_pred, X_test, level=level)
                    elif scoring == "coverage":
                        return coverage(X_pred, X_test, level=level)
                    else:
                        return mean_errors(
                            pred=X_pred.mean, actual=X_test, scoring=scoring
                        )
                else:  # not probabilistic
                    return mean_errors(
                        pred=X_pred, actual=X_test, scoring=scoring
                    )

        else:  # isinstance(scoring, str) = False
            err_func = scoring

        errors = []

        train_indices = []

        test_indices = []

        for train_index, test_index in tscv_obj:
            train_indices.append(train_index)
            test_indices.append(test_index)

        if show_progress is True:
            iterator = tqdm(
                zip(train_indices, test_indices), total=len(train_indices)
            )
        else:
            iterator = zip(train_indices, test_indices)

        for train_index, test_index in iterator:
            if verbose == 1:
                print(f"TRAIN: {train_index}")
                print(f"TEST: {test_index}")

            if isinstance(X, pd.DataFrame):
                self.fit(X.iloc[train_index, :], xreg=xreg, **kwargs)
                X_test = X.iloc[test_index, :]
            else:
                self.fit(X[train_index, :], xreg=xreg, **kwargs)
                X_test = X[test_index, :]
            X_pred = self.predict(h=int(len(test_index)), level=level, **kwargs)

            errors.append(err_func(X_test, X_pred, scoring, alpha=alpha))

        res = np.asarray(errors)

        return res, describe(res)

    def _compute_information_criterion(self, curr_lags, criterion="AIC"):
        """Compute information criterion using existing residuals

        Parameters
        ----------
        curr_lags : int
            Current number of lags being evaluated
        criterion : str
            One of 'AIC', 'AICc', or 'BIC'

        Returns
        -------
        float
            Information criterion value or inf if parameters exceed observations
        """
        # Get dimensions
        n_obs = self.residuals_.shape[0]
        n_features = int(self.init_n_series_ * curr_lags)
        n_hidden = int(self.n_hidden_features)
        # Calculate number of parameters
        term1 = int(n_features * n_hidden)
        term2 = int(n_hidden * self.init_n_series_)
        n_params = term1 + term2
        # Check if we have enough observations for the number of parameters
        if n_obs <= n_params + 1:
            return float("inf")  # Return infinity if too many parameters
        # Compute RSS using existing residuals
        rss = np.sum(self.residuals_**2)
        # Compute criterion
        if criterion == "AIC":
            ic = n_obs * np.log(rss / n_obs) + 2 * n_params
        elif criterion == "AICc":
            ic = n_obs * np.log(rss / n_obs) + 2 * n_params * (
                n_obs / (n_obs - n_params - 1)
            )
        else:  # BIC
            ic = n_obs * np.log(rss / n_obs) + n_params * np.log(n_obs)

        return ic
