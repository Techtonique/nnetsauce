# Authors: Thierry Moudiki
#
# License: BSD 3 Clear Clause

import copy
import numpy as np
import pandas as pd
import sklearn.metrics as skm2
import matplotlib.pyplot as plt
from collections import namedtuple
from copy import deepcopy
from functools import partial
from scipy.stats import describe, norm
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
)
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
from tqdm import tqdm
from ..base import Base
from ..sampling import vinecopula_sample
from ..simulation import getsims
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

        type_pi: str.
            type of prediction interval; currently:
            - "gaussian": simple, fast, but: assumes stationarity of Gaussian in-sample residuals and independence in the multivariate case
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
        block_size=None,
        replications=None,
        kernel="gaussian",
        agg="mean",
        seed=123,
        backend="cpu",
        verbose=0,
        show_progress=True,
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
        self.type_pi = type_pi
        self.block_size = block_size
        self.replications = replications
        self.kernel = kernel
        self.agg = agg
        self.verbose = verbose
        self.show_progress = show_progress
        self.series_names = None
        self.input_dates = None
        self.fit_objs_ = {}
        self.y_ = None  # MTS responses (most recent observations first)
        self.X_ = None  # MTS lags
        self.xreg_ = None
        self.y_means_ = {}
        self.mean_ = None
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

        if (
            isinstance(X, pd.DataFrame) is False
        ):  # input data set is a numpy array

            if xreg is None:
                X = pd.DataFrame(X)
                self.series_names = [
                    "series" + str(i) for i in range(X.shape[1])
                ]
            else:  # xreg is not None
                X = mo.cbind(X, xreg)
                self.xreg_ = xreg

        else:  # input data set is a DataFrame with column names

            #if "date" in X.columns: 
            #    X.index = X["date"]
            #    X.drop(['date'], axis=1, inplace=True)

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
            self.df_ = X
            X = X.values
            self.df_.columns = self.series_names
            self.input_dates = ts.compute_input_dates(self.df_)
        else:
            self.df_ = pd.DataFrame(X, columns=self.series_names)
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

        dummy_y, scaled_Z = self.cook_training_set(y=rep_1_n, X=self.X_)

        self.scaled_Z_ = scaled_Z

        # loop on all the time series and adjust self.obj.fit
        if self.verbose > 0:
            print(
                f"\n Adjusting {type(self.obj).__name__} to multivariate time series... \n "
            )

        if self.show_progress is True:
            iterator = tqdm(range(p))
        else:
            iterator = range(p)

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

    def predict(self, h=5, level=95, **kwargs):
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

        self.output_dates_, frequency = ts.compute_output_dates(self.df_, h)

        self.level_ = level

        self.return_std_ = False  # do not remove (/!\)

        self.mean_ = None  # do not remove (/!\)

        self.mean_ = deepcopy(self.y_)  # do not remove (/!\)

        self.lower_ = None  # do not remove (/!\)

        self.upper_ = None  # do not remove (/!\)

        self.sims_ = None  # do not remove (/!\)

        y_means_ = np.asarray([self.y_means_[i] for i in range(self.n_series)])

        n_features = self.n_series * self.lags

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
            DescribeResult = namedtuple(
                "DescribeResult", ("mean", "lower", "upper")
            )  # to be updated

        if self.kde_ != None and "kde" in self.type_pi:  # kde
            if self.verbose == 1:
                self.residuals_sims_ = tuple(
                    self.kde_.sample(
                        n_samples=h, random_state=self.seed + 100 * i
                    )
                    for i in tqdm(range(self.replications))
                )
            elif self.verbose == 0:
                self.residuals_sims_ = tuple(
                    self.kde_.sample(
                        n_samples=h, random_state=self.seed + 100 * i
                    )
                    for i in range(self.replications)
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

        for _ in range(h):

            new_obs = ts.reformat_response(self.mean_, self.lags)

            new_X = new_obs.reshape(1, n_features)

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
                    preds_pi = self.fit_objs_[i].predict(
                        cooked_new_X, return_pi=True
                    )
                    mean_pi_.append(preds_pi.mean[0])
                    lower_pi_.append(preds_pi.lower[0])
                    upper_pi_.append(preds_pi.upper[0])

            predicted_cooked_new_X = np.asarray(
                [
                    np.asarray(self.fit_objs_[i].predict(cooked_new_X)).item()
                    for i in range(self.n_series)
                ]
            )

            preds = np.asarray(y_means_ + predicted_cooked_new_X)

            self.mean_ = mo.rbind(preds, self.mean_)  # preallocate?

        # function's return ----------------------------------------------------------------------
        self.mean_ = pd.DataFrame(
            self.mean_[0:h, :][::-1],
            columns=self.df_.columns,
            index=self.output_dates_,
        )

        if (
            (("return_std" not in kwargs) and ("return_pi" not in kwargs))
            and (self.type_pi not in ("gaussian", "scp"))
        ) or ("vine" in self.type_pi):

            if self.replications is None:
                return self.mean_

            # if "return_std" not in kwargs and self.replications is not None
            meanf = []
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
            for ix in range(self.n_series):
                sims_ix = getsims(self.sims_, ix)
                if self.agg == "mean":
                    meanf.append(np.mean(sims_ix, axis=1))
                else:
                    meanf.append(np.median(sims_ix, axis=1))
                lower.append(np.quantile(sims_ix, q=self.alpha_ / 200, axis=1))
                upper.append(
                    np.quantile(sims_ix, q=1 - self.alpha_ / 200, axis=1)
                )

            self.mean_ = pd.DataFrame(
                np.asarray(meanf).T,
                columns=self.series_names,  # self.df_.columns,
                index=self.output_dates_,
            )

            self.lower_ = pd.DataFrame(
                np.asarray(lower).T,
                columns=self.series_names,  # self.df_.columns,
                index=self.output_dates_,
            )

            self.upper_ = pd.DataFrame(
                np.asarray(upper).T,
                columns=self.series_names,  # self.df_.columns,
                index=self.output_dates_,
            )

            res = DescribeResult(
                self.mean_, self.sims_, self.lower_, self.upper_
            )

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

                return res2

            else:

                return res

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

            self.lower_ = pd.DataFrame(
                self.mean_.values - pi_multiplier * self.gaussian_preds_std_,
                columns=self.series_names,  # self.df_.columns,
                index=self.output_dates_,
            )

            self.upper_ = pd.DataFrame(
                self.mean_.values + pi_multiplier * self.gaussian_preds_std_,
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

    def score(self, X, training_index, testing_index, scoring=None, **kwargs):
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

        # if p > 1:
        #     return tuple(
        #         [
        #             scoring_options[scoring](
        #                 X_test[:, i], preds[:, i]#, **kwargs
        #             )
        #             for i in range(p)
        #         ]
        #     )
        # else:
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
            assert (
                self.n_series == 1
            ), "please specify series index or name (n_series > 1)"
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
                "root_mean_squared_error",
                "mean_squared_error",
                "mean_error",
                "mean_absolute_error",
                "mean_percentage_error",
                "mean_absolute_percentage_error",
                "winkler_score",
                "coverage",
            ), "must have scoring in ('root_mean_squared_error', 'mean_squared_error', 'mean_error', 'mean_absolute_error', 'mean_error', 'mean_percentage_error', 'mean_absolute_percentage_error',  'winkler_score', 'coverage')"

            def err_func(X_test, X_pred, scoring):
                if (self.replications is not None) or (
                    self.type_pi == "gaussian"
                ):  # probabilistic
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

            errors.append(err_func(X_test, X_pred, scoring))

        res = np.asarray(errors)

        return res, describe(res)
