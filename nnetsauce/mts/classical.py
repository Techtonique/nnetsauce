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
    mean_absolute_error,
    mean_absolute_percentage_error,
)
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
try: 
    from statsmodels.tsa.api import VAR
    from statsmodels.tsa.vector_ar.vecm import VECM
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from statsmodels.tsa.forecasting.theta import ThetaModel
except ImportError:
    pass

from tqdm import tqdm
from ..base import Base
from ..sampling import vinecopula_sample
from ..simulation import getsims
from ..utils import matrixops as mo
from ..utils import misc as mx
from ..utils import timeseries as ts
from ..utils import convert_df_to_numeric, coverage, winkler_score, mean_errors
from ..utils import TimeSeriesSplit


class ClassicalMTS(Base):
    """Multivariate time series (FactorMTS) forecasting with Factor models

    Parameters:

        model: type of model: str.
            currently, 'VAR', 'VECM', 'ARIMA', 'ETS', 'Theta'

    Attributes:

        df_: data frame
            the input data frame, in case a data.frame is provided to `fit`

        level_: int
            level of confidence for prediction intervals (default is 95)

    Examples:
    See examples/classical_mts_timeseries.py
    """

    # construct the object -----

    def __init__(self, model="VAR"):

        self.model = model
        if self.model == "VAR":
            self.obj = VAR
        elif self.model == "VECM":
            self.obj = VECM
        elif self.model == "ARIMA":
            self.obj = ARIMA
        elif self.model == "ETS":
            self.obj = ExponentialSmoothing
        elif self.model == "Theta":
            self.obj = ThetaModel
        else:
            raise ValueError("model not recognized")
        self.n_series = None
        self.replications = None
        self.mean_ = None
        self.upper_ = None
        self.lower_ = None
        self.output_dates_ = None
        self.alpha_ = None
        self.df_ = None
        self.residuals_ = []
        self.sims_ = None
        self.level_ = None

    def fit(self, X, **kwargs):
        """Fit FactorMTS model to training data X, with optional regressors xreg

        Parameters:

        X: {array-like}, shape = [n_samples, n_features]
            Training time series, where n_samples is the number
            of samples and n_features is the number of features;
            X must be in increasing order (most recent observations last)

        **kwargs: for now, additional parameters to be passed to for kernel density estimation, when needed (see sklearn.neighbors.KernelDensity)

        Returns:

        self: object
        """

        try:
            self.n_series = X.shape[1]        
        except Exception:
            self.n_series = 1

        if (
            isinstance(X, pd.DataFrame) is False
        ) and isinstance(X, pd.Series) is False :  # input data set is a numpy array
            
            X = pd.DataFrame(X)
            if self.n_series > 1: 
                self.series_names = ["series" + str(i) for i in range(X.shape[1])]
            else: 
                self.series_names = "series0"

        else:  # input data set is a DataFrame or Series with column names
            
            X_index = None
            if X.index is not None and len(X.shape) > 1:
                X_index = X.index
                X = copy.deepcopy(mo.convert_df_to_numeric(X))
            if X_index is not None:
                try:
                    X.index = X_index
                except Exception:
                    pass
            if isinstance(X, pd.DataFrame): 
                self.series_names = X.columns.tolist()
            else:
                self.series_names = X.name

        if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
            self.df_ = X
            X = X.values
            self.df_.columns = self.series_names
            self.input_dates = ts.compute_input_dates(self.df_)
        else:
            self.df_ = pd.DataFrame(X, columns=self.series_names)            

        if self.model == "Theta":
            self.obj = self.obj(self.df_, 
                                **kwargs).fit()
        else:        
            self.obj = self.obj(X, **kwargs).fit(**kwargs)        

        return self

    def predict(self, h=5, level=95, **kwargs):
        """Forecast all the time series, h steps ahead

        Parameters:

        h: {integer}
            Forecasting horizon

        **kwargs: additional parameters to be passed to
                self.cook_test_set

        Returns:

        model predictions for horizon = h: {array-like}

        """

        self.output_dates_, frequency = ts.compute_output_dates(self.df_, h)

        self.level_ = level

        self.lower_ = None  # do not remove (/!\)

        self.upper_ = None  # do not remove (/!\)

        self.sims_ = None  # do not remove (/!\)

        self.level_ = level

        self.alpha_ = 100 - level
        
        pi_multiplier = norm.ppf(1 - self.alpha_ / 200)

        # Named tuple for forecast results
        DescribeResult = namedtuple(
            "DescribeResult", ("mean", "lower", "upper")
        )

        if self.model == "VAR":
            mean_forecast, lower_bound, upper_bound = (
                self.obj.forecast_interval(
                    self.obj.endog, steps=h, alpha=self.alpha_ / 100, **kwargs
                )
            )

        elif self.model == "VECM":
            forecast_result = self.obj.predict(steps=h)
            mean_forecast = forecast_result
            lower_bound, upper_bound = self._compute_confidence_intervals(
                forecast_result, alpha=self.alpha_ / 100, **kwargs
            )
        
        elif self.model == "ARIMA":
            forecast_result = self.obj.get_forecast(steps=h)
            mean_forecast = forecast_result.predicted_mean
            lower_bound = forecast_result.conf_int()[:, 0]
            upper_bound = forecast_result.conf_int()[:, 1]    
        
        elif self.model == "ETS":
            forecast_result = self.obj.forecast(steps=h)
            residuals = self.obj.resid
            std_errors = np.std(residuals)
            mean_forecast = forecast_result
            lower_bound = forecast_result - pi_multiplier * std_errors
            upper_bound = forecast_result + pi_multiplier * std_errors
        
        elif self.model == "Theta":
            try:
                mean_forecast = self.obj.forecast(steps=h).values 
                forecast_result = self.obj.prediction_intervals(steps=h, alpha=self.alpha_/100, 
                                                            **kwargs)            
                lower_bound = forecast_result["lower"].values 
                upper_bound = forecast_result["upper"].values 
            except Exception: 
                mean_forecast = self.obj.forecast(steps=h)
                forecast_result = self.obj.prediction_intervals(steps=h, alpha=self.alpha_/100, 
                                                            **kwargs)            
                lower_bound = forecast_result["lower"]
                upper_bound = forecast_result["upper"]
            
        else:

            raise ValueError("model not recognized")                    

        try: 
            self.mean_ = pd.DataFrame(mean_forecast, 
                                        columns=self.series_names, 
                                        index=self.output_dates_)        
            self.lower_ = pd.DataFrame(lower_bound, 
                                    columns=self.series_names, 
                                        index=self.output_dates_)
            self.upper_ = pd.DataFrame(upper_bound, 
                                    columns=self.series_names, 
                                        index=self.output_dates_)
        except Exception:
            self.mean_ = pd.Series(mean_forecast, 
                                    name=self.series_names, 
                                    index=self.output_dates_)        
            self.lower_ = pd.Series(lower_bound, 
                                    name=self.series_names, 
                                    index=self.output_dates_)
            self.upper_ = pd.Series(upper_bound, 
                                    name=self.series_names, 
                                    index=self.output_dates_)
        

        return DescribeResult(
            mean=self.mean_, lower=self.lower_, upper=self.upper_
        )

    def _compute_confidence_intervals(self, forecast_result, alpha):
        """
        Compute confidence intervals for VECM forecasts.
        Uses the covariance of residuals to approximate the confidence intervals.
        """
        residuals = self.obj.resid
        cov_matrix = np.cov(residuals.T)  # Covariance matrix of residuals
        std_errors = np.sqrt(np.diag(cov_matrix))  # Standard errors

        z_value = norm.ppf(1 - alpha / 2)  # Z-score for the given alpha level
        lower_bound = forecast_result - z_value * std_errors
        upper_bound = forecast_result + z_value * std_errors

        return lower_bound, upper_bound

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
            "neg_mean_squared_error": lambda x, y: np.mean((x - y)**2),
            "neg_root_mean_squared_error": lambda x, y: np.sqrt(np.mean((x - y)**2)),
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

        if isinstance(self.df_, pd.DataFrame): 
            y_all = list(self.df_.iloc[:, series_idx]) + list(
                self.mean_.iloc[:, series_idx]
            )
            y_test = list(self.mean_.iloc[:, series_idx])
        else:
            y_all = list(self.df_.values) + list(
                self.mean_.values
            )
            y_test = list(self.mean_.values)
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
            try: 
                ax.fill_between(
                    x_test,
                    self.lower_.iloc[:, series_idx],
                    self.upper_.iloc[:, series_idx],
                    alpha=0.2,
                    color="orange",
                )
            except Exception:
                ax.fill_between(
                    x_test,
                    self.lower_.values,
                    self.upper_.values,
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
