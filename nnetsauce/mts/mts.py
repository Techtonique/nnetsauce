# Authors: Thierry Moudiki
#
# License: BSD 3 Clear Clause

import numpy as np
import pandas as pd
import pickle
import sklearn.metrics as skm2
import matplotlib.pyplot as plt 
from collections import namedtuple
from datetime import datetime
from scipy.stats import norm, describe
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
from tqdm import tqdm
from ..base import Base
from ..simulation import getsims
from ..utils import matrixops as mo
from ..utils import timeseries as ts


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
        replications=None,
        kernel=None,
        agg="mean",
        seed=123,
        backend="cpu",
        verbose=0
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
            backend=backend
        )

        self.obj = obj
        self.n_series = None
        self.lags = lags
        self.replications = replications
        self.kernel = kernel 
        self.agg = agg
        self.verbose = verbose
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
        self.alpha_ = None
        self.return_std_ = None
        self.df_ = None
        self.residuals_ = [] 
        self.residuals_sims_ = None
        self.kde_ = None
        self.sims_ = None

    def fit(self, X, xreg=None, **kwargs):
        """Fit MTS model to training data X, with optional regressors xreg

        Parameters:

            X: {array-like}, shape = [n_samples, n_features]
                Training time series, where n_samples is the number
                of samples and n_features is the number of features;
                X must be in increasing order (most recent observations last)

            xreg: {array-like}, shape = [n_samples, n_features_xreg]
                Additional regressors to be passed to obj
                xreg must be in increasing order (most recent observations last)

            **kwargs: for now, additional parameters to be passed to for kernel density estimation, when needed (see sklearn.neighbors.KernelDensity)

        Returns:

            self: object
        """

        if isinstance(X, pd.DataFrame) is False:
            self.series_names = ["series" + str(i) for i in range(X.shape[1])]
            X = pd.DataFrame(X, columns=self.series_names)
        else:
            self.series_names = list(X.columns.values)
        
        self.df_ = X
        X = X.values 
        self.input_dates = ts.compute_input_dates(self.df_)

        # Backup input dates --> ts.compute_input_dates(self.df_)
        """
        input_dates = self.df_.index.values
        print(f"input_dates 1: {input_dates}")
        frequency = pd.infer_freq(pd.DatetimeIndex(input_dates))
        print(f"frequency: {frequency}")
        input_dates = pd.date_range(
            start=input_dates[0], periods=len(input_dates), freq=frequency
        ).values.tolist()
        print(f"input_dates 2: {input_dates}")
        df_input_dates = pd.DataFrame({"date": input_dates})
        print(f"input_dates 3: {df_input_dates}")
        self.input_dates = pd.to_datetime(df_input_dates["date"]).dt.date
        print(f"input_dates 4: {self.input_dates}")
        """              

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
        residuals_ = []
        self.residuals_ = None
        self.residuals_sims_ = None
        self.kde_ = None 
        self.sims_ = None

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
        if self.verbose > 0:
            print(f"\n Adjusting {type(self.obj).__name__} to multivariate time series... \n ")

        for i in tqdm(range(p)):
            y_mean = np.mean(self.y_[:, i])
            self.y_means_[i] = y_mean
            centered_y_i = self.y_[:, i] - y_mean
            self.obj.fit(X = scaled_Z, y = centered_y_i)
            self.fit_objs_[i] = pickle.loads(pickle.dumps(self.obj, -1))
            residuals_.append((centered_y_i - self.fit_objs_[i].predict(scaled_Z)).tolist())

        self.residuals_ = np.asarray(residuals_).T
                        
        if self.replications is not None:
            if self.verbose > 0:
                print(f"\n Simulate residuals using {self.kernel} kernel... \n")
            assert self.kernel in ('gaussian', 'tophat'), "currently, 'kernel' must be either 'gaussian' or 'tophat'"
            kernel_bandwidths = {"bandwidth": np.logspace(-6, 6, 100)}
            
            grid = GridSearchCV(KernelDensity(kernel = self.kernel, **kwargs), 
                                param_grid=kernel_bandwidths)  
            
            grid.fit(self.residuals_) 
            
            if self.verbose > 0:
                print(f"\n Best parameters for {self.kernel} kernel: {grid.best_params_} \n")

            self.kde_ = grid.best_estimator_
                
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

        self.output_dates_, frequency = ts.compute_output_dates(self.df_, h)        

       # print(f" \n self.output_dates_: {self.output_dates_} \n ")

        self.return_std_ = False # do not remove (/!\)

        self.mean_ = None  # do not remove (/!\)

        self.mean_ = pickle.loads(pickle.dumps(self.y_, -1))

        self.lower_ = None

        self.upper_ = None

        self.sims_ = None

        y_means_ = np.asarray([self.y_means_[i] for i in range(self.n_series)])

        n_features = self.n_series * self.lags        

        self.alpha_ = 100 - level

        if "return_std" in kwargs:
            self.return_std_ = True 
            self.preds_std_ = []            
            pi_multiplier = norm.ppf(1 - self.alpha_/200)
            DescribeResult = namedtuple('DescribeResult', ('mean', 'lower', 'upper')) # to be updated

        if self.xreg_ is None: # no external regressors 

            if self.kde_ is not None: 
                self.residuals_sims_ = tuple(self.kde_.sample(n_samples = h, 
                                                              random_state=self.seed + 100*i) for i in tqdm(range(self.replications)))

            for _ in range(h):

                new_obs = ts.reformat_response(self.mean_, self.lags)

                new_X = new_obs.reshape(1, n_features)

                cooked_new_X = self.cook_test_set(new_X, **kwargs)

                if "return_std" in kwargs:
                    self.preds_std_.append([np.asarray(self.fit_objs_[i].predict(cooked_new_X, return_std=True)[1]).item() for i in range(self.n_series)])

                predicted_cooked_new_X = np.asarray([np.asarray(self.fit_objs_[i].predict(
                    cooked_new_X)).item() for i in range(self.n_series)])

                preds = np.asarray(y_means_ + predicted_cooked_new_X)                                                                     
                
                self.mean_ = mo.rbind(preds, self.mean_)
        
        else: # if self.xreg_ is not None: # with external regressors
            
            assert new_xreg is not None, "'new_xreg' must be provided to predict()"

            if isinstance(new_xreg, pd.DataFrame):
                new_xreg = new_xreg.values

            if self.kde_ is not None: 
                if self.verbose > 0:
                    print("\n Obtain simulations for adjusted residuals... \n")
                self.residuals_sims_ = tuple(self.kde_.sample(n_samples = h, random_state=self.seed + 100*i) for i in tqdm(range(self.replications)))

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

            for _ in range(h):

                new_obs = ts.reformat_response(self.mean_, self.lags)

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
                
                self.mean_ = mo.rbind(preds, self.mean_)        
        
        # function's return ----------------------------------------------------------------------
        self.mean_ = pd.DataFrame(
            self.mean_[0:h, :][::-1],
            columns=self.df_.columns,
            index=self.output_dates_
        )
        if "return_std" not in kwargs:
            if self.kde_ is None: 
                return self.mean_  

            # if "return_std" not in kwargs and self.kde_ is not None  
            meanf = []
            lower = []
            upper = []
            self.sims_ = tuple((self.mean_ + self.residuals_sims_[i] for i in tqdm(range(self.replications))))                                           
            DescribeResult = namedtuple('DescribeResult', 
                                        ('mean', 'sims', 'lower', 'upper'))                         
            for ix in range(self.n_series):
                sims_ix = getsims(self.sims_, ix)
                if self.agg == "mean":
                    meanf.append(np.mean(sims_ix, axis=1))
                else:
                    meanf.append(np.median(sims_ix, axis=1))
                lower.append(np.quantile(sims_ix, q = self.alpha_/200, axis=1))
                upper.append(np.quantile(sims_ix, q = 1 - self.alpha_/200, axis=1))
            
            self.mean_ = pd.DataFrame(np.asarray(meanf).T,
                                      columns=self.df_.columns,
                                      index=self.output_dates_)

            self.lower_ = pd.DataFrame(np.asarray(lower).T,
                                 columns=self.df_.columns,
                                 index=self.output_dates_)

            self.upper_ = pd.DataFrame(np.asarray(upper).T,
                                 columns=self.df_.columns,
                                 index=self.output_dates_)

            return DescribeResult(self.mean_, 
                                  self.sims_, 
                                  self.lower_, 
                                  self.upper_)       
            
        # if "return_std" in kwargs
        DescribeResult = namedtuple('DescribeResult', 
                                    ('mean', 'lower', 'upper')) 
        
        self.mean_ = pd.DataFrame(
            np.asarray(self.mean_),
            columns=self.df_.columns,
            index=self.output_dates_,
        )

        self.preds_std_ = np.asarray(self.preds_std_)

        self.lower_ = pd.DataFrame(self.mean_.values-pi_multiplier*self.preds_std_,
                                   columns=self.df_.columns,
                                   index=self.output_dates_)

        self.upper_ = pd.DataFrame(self.mean_.values+pi_multiplier*self.preds_std_,
                                   columns=self.df_.columns,
                                   index=self.output_dates_)

        return DescribeResult(self.mean_, 
                              self.lower_, 
                              self.upper_)       


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


    def plot(self, series):
        """Plot time series forecast 

        Parameters:

            series: {integer} or {string}
                series index or name 
        """

        type_graph = "dates" # not clean, stabilize this in future releases

        assert all([self.mean_ is not None, self.lower_ is not None, 
                    self.upper_ is not None, self.output_dates_ is not None]),\
                    "model forecasting must be obtained first (with predict)"

        if isinstance(series, str):
            assert series in self.series_names, f"series {series} doesn't exist in the input dataset"
            series_idx = self.df_.columns.get_loc(series)
        else:
            assert isinstance(series, int) and (0 <= series < self.n_series),\
                  f"check series index (< {self.n_series})"
            series_idx = series

        y_all = list(self.df_.iloc[:, series_idx])+list(self.mean_.iloc[:, series_idx])
        y_test = list(self.mean_.iloc[:, series_idx])
        n_points_all = len(y_all)
        n_points_train = self.df_.shape[0]

        if type_graph is "numeric":
            x_all = [i for i in range(n_points_all)]
            x_test = [i for i in range(n_points_train, n_points_all)]              
        else: # use dates       
            x_all = np.concatenate((self.input_dates.values, self.output_dates_.values), axis=None)
            x_test = self.output_dates_.values        

        fig, ax = plt.subplots()
        ax.plot(x_all, y_all, '-')
        ax.plot(x_test, y_test, '-', color = "orange")
        ax.fill_between(x_test, self.lower_.iloc[:, series_idx], 
                        self.upper_.iloc[:, series_idx], 
                        alpha=0.2, color = "orange")
        plt.show()
