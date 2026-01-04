import copy
import numpy as np
import pandas as pd

from .mts import MTS
from ..utils import matrixops as mo
from ..utils import timeseries as ts
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity
from tqdm import tqdm

class MultiOutputMTS(MTS):
    """MTS subclass optimized for multivariate time series with vectorized models
    
    Enforces n_series >= 2 and uses single vectorized fit call instead of per-series loop.
    Works with sklearn models supporting multi-output (Ridge, Lasso, LinearRegression, etc.)
    """
    
    def fit(self, X, xreg=None, **kwargs):
        """Fit with vectorized multi-output model - requires n_series >= 2"""
        
        # Validate multivariate input
        try:
            self.init_n_series_ = X.shape[1]
        except IndexError:
            raise ValueError("MultiOutputMTS requires multivariate input (n_samples, n_series)")
        
        if self.init_n_series_ < 2:
            raise ValueError(f"MultiOutputMTS requires at least 2 series, got {self.init_n_series_}")
        
        # Automatic lag selection if requested (copied from parent)
        if isinstance(self.lags, str):
            max_lags = min(25, X.shape[0] // 4)
            best_ic = float("inf")
            best_lags = 1
            
            if self.verbose:
                print(f"\nSelecting optimal number of lags using {self.lags}...")
                iterator = tqdm(range(1, max_lags + 1))
            else:
                iterator = range(1, max_lags + 1)
            
            for lag in iterator:
                if isinstance(X, pd.DataFrame):
                    X_values = X.values[::-1]
                else:
                    X_values = X[::-1]
                
                mts_input = ts.create_train_inputs(X_values, lag)
                dummy_y, scaled_Z = self.cook_training_set(
                    y=np.ones(mts_input[0].shape[0]), X=mts_input[1]
                )
                
                # Vectorized fit for lag selection
                y_means = np.mean(mts_input[0], axis=0)
                centered_y = mts_input[0] - y_means[np.newaxis, :]
                self.obj.fit(X=scaled_Z, y=centered_y)
                residuals = centered_y - self.obj.predict(scaled_Z)
                self.residuals_ = residuals  # Keep (n_obs, n_series) shape
                
                ic = self._compute_information_criterion(curr_lags=lag, criterion=self.lags)
                
                if self.verbose:
                    print(f"Trying lags={lag}, {self.lags}={ic:.2f}")
                
                if ic < best_ic:
                    best_ic = ic
                    best_lags = lag
            
            if self.verbose:
                print(f"\nSelected {best_lags} lags with {self.lags}={best_ic:.2f}")
            
            self.lags = best_lags
        
        # Data preprocessing (from parent)
        self.input_dates = None
        self.df_ = None
        
        if isinstance(X, pd.DataFrame) is False:
            if xreg is None:
                X = pd.DataFrame(X)
                self.series_names = ["series" + str(i) for i in range(X.shape[1])]
            else:
                X = mo.cbind(X, xreg)
                self.xreg_ = xreg
        else:
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
        
        n, p = X.shape
        self.n_obs_ = n
        rep_1_n = np.repeat(1, n)
        
        self.y_ = None
        self.X_ = None
        self.n_series = p
        self.fit_objs_.clear()
        self.y_means_.clear()
        self.residuals_ = None
        self.residuals_sims_ = None
        self.kde_ = None
        self.sims_ = None
        self.scaled_Z_ = None
        self.centered_y_is_ = []
        
        # Create training inputs
        mts_input = ts.create_train_inputs(X[::-1], self.lags)
        self.y_ = mts_input[0]
        self.X_ = mts_input[1]
        
        dummy_y, scaled_Z = self.cook_training_set(y=rep_1_n, X=self.X_)
        self.scaled_Z_ = scaled_Z
        
        if self.verbose > 0:
            print(f"\n Adjusting {type(self.obj).__name__} to multivariate time series (vectorized)... \n")
        
        # VECTORIZED FITTING - NO LOOP
        y_means_array = np.array([np.mean(self.y_[:, i]) for i in range(self.init_n_series_)])
        for i in range(self.init_n_series_):
            self.y_means_[i] = y_means_array[i]
        
        centered_y_all = self.y_ - y_means_array[np.newaxis, :]
        self.centered_y_is_ = [centered_y_all[:, i] for i in range(self.init_n_series_)]
        
        # Single vectorized fit for all series
        self.obj.fit(scaled_Z, centered_y_all)
        
        # All series share the same model
        for i in range(self.init_n_series_):
            self.fit_objs_[i] = self.obj
        
        # Vectorized residuals - ONLY target columns (n_obs, n_series)
        preds_all = self.obj.predict(scaled_Z)
        residuals_raw = centered_y_all - preds_all
        
        # CRITICAL: Ensure residuals only have n_series columns, not all scaled_Z columns
        # In case there's some dimension mismatch, explicitly slice
        self.residuals_ = residuals_raw[:, :self.init_n_series_]
        
        # Handle type_pi
        if self.type_pi == "gaussian":
            self.gaussian_preds_std_ = np.std(self.residuals_, axis=0)
        
        if self.type_pi.startswith("scp2"):
            data_mean = np.mean(self.residuals_, axis=0)
            self.residuals_std_dev_ = np.std(self.residuals_, axis=0)
            self.residuals_ = (self.residuals_ - data_mean[np.newaxis, :]) / self.residuals_std_dev_[np.newaxis, :]
        
        if self.replications is not None and "kde" in self.type_pi:
            if self.verbose > 0:
                print(f"\n Simulate residuals using {self.kernel} kernel... \n")
            assert self.kernel in ("gaussian", "tophat"), "currently, 'kernel' must be either 'gaussian' or 'tophat'"
            kernel_bandwidths = {"bandwidth": np.logspace(-6, 6, 150)}
            grid = GridSearchCV(
                KernelDensity(kernel=self.kernel, **kwargs),
                param_grid=kernel_bandwidths,
            )
            grid.fit(self.residuals_)
            if self.verbose > 0:
                print(f"\n Best parameters for {self.kernel} kernel: {grid.best_params_} \n")
            self.kde_ = grid.best_estimator_
        
        return self
    
    def predict(self, h=5, level=95, quantiles=None, **kwargs):
        """Override predict to handle vectorized model predictions"""
        
        # Delegate to parent for quantiles and multiple levels
        if quantiles is not None or isinstance(level, (list, np.ndarray)):
            return super().predict(h=h, level=level, quantiles=quantiles, **kwargs)
        
        # Store original obj temporarily
        original_obj = self.obj
        
        # Create wrapper that extracts the i-th output for each series
        class VectorizedWrapper:
            def __init__(self, model, series_idx):
                self.model = model
                self.series_idx = series_idx
            
            def predict(self, X, **kw):
                """Predict and return only the output for this series index"""
                preds = self.model.predict(X, **kw)
                # preds shape: (n_samples, n_series) or (n_series,)
                if len(preds.shape) == 1:
                    # Single prediction: (n_series,)
                    return preds[self.series_idx:self.series_idx+1]
                else:
                    # Multiple predictions: (n_samples, n_series)
                    return preds[:, self.series_idx:self.series_idx+1].flatten()
        
        # Wrap each series with its own index
        for i in range(self.init_n_series_):
            self.fit_objs_[i] = VectorizedWrapper(original_obj, i)
        
        try:
            result = super().predict(h=h, level=level, quantiles=quantiles, **kwargs)
        finally:
            # Restore original
            for i in range(self.init_n_series_):
                self.fit_objs_[i] = original_obj
        
        return result