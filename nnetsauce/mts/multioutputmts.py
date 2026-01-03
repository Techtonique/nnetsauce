import warnings
import numpy as np
import pandas as pd
from copy import deepcopy
from tqdm import tqdm
from sklearn.base import clone
from sklearn.utils.validation import check_X_y
from scipy.stats import norm

from ..mts import MTS 
from ..sampling import vinecopula_sample
from ..simulation import getsims, getsimsxreg
from ..quantile import QuantileRegressor
from ..utils import matrixops as mo
from ..utils import misc as mx
from ..utils import timeseries as ts
from ..utils import convert_df_to_numeric


class MultiOutputMTS(MTS):
    """Multivariate time series forecasting with multioutput regression support.
    
    This class extends MTS to efficiently handle multioutput regressors
    (e.g., Ridge, LinearRegression) by fitting all series simultaneously
    without loops, for better performance.
    
    Parameters:
        obj: object
            A multioutput regressor containing methods fit() and predict()
            that support multioutput regression (y can be 2D).
            
        force_multioutput: bool, default=False
            If True, force multioutput mode even for univariate series.
            If False, automatically falls back to parent MTS for univariate.
            
        All other parameters are inherited from MTS.
    
    Notes:
        - The base object must support multioutput regression (accept 2D y in fit)
        - Not compatible with type_pi that require special handling (e.g., SCP variants)
        - For quantile regression, ensure obj supports multioutput or use MultiOutputRegressor wrapper
        - Automatically tests multioutput capability during initialization
    
    Examples:
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from sklearn.linear_model import Ridge
    >>> from sklearn.ensemble import RandomForestRegressor
    >>> from sklearn.multioutput import MultiOutputRegressor
    >>> import nnetsauce as ns
    >>>
    >>> # Generate sample multivariate time series
    >>> np.random.seed(123)
    >>> n_samples, n_series = 100, 3
    >>> base_series = np.cumsum(np.random.randn(n_samples))
    >>> series_data = np.zeros((n_samples, n_series))
    >>> for i in range(n_series):
    >>>     series_data[:, i] = base_series + np.cumsum(np.random.randn(n_samples) * 0.5) + i * 10
    >>> 
    >>> df = pd.DataFrame(series_data, columns=[f'series_{i}' for i in range(n_series)])
    >>>
    >>> # Example 1: Ridge regression (native multioutput support)
    >>> ridge = Ridge(alpha=1.0)
    >>> model = ns.MultiOutputMTS(
    >>>     obj=ridge,
    >>>     lags=3,
    >>>     n_hidden_features=10,
    >>>     type_pi='kde',
    >>>     replications=100,
    >>>     verbose=1
    >>> )
    >>> model.fit(df)
    >>> forecast = model.predict(h=10, level=95)
    >>> print(f"Forecast shape: {forecast.mean.shape}")
    >>> print(f"Multioutput used: {model.get_multioutput_info()['multioutput_used']}")
    >>>
    >>> # Example 2: RandomForest with wrapper
    >>> rf = RandomForestRegressor(n_estimators=50)
    >>> multi_rf = MultiOutputRegressor(rf)
    >>> model_rf = ns.MultiOutputMTS(
    >>>     obj=multi_rf,
    >>>     lags=3,
    >>>     type_pi='gaussian',
    >>>     verbose=0
    >>> )
    >>> model_rf.fit(df)
    >>> forecast_rf = model_rf.predict(h=10, level=90)
    """
    
    def __init__(self, obj, force_multioutput=False, **kwargs):
        # Store original object
        self._original_obj = obj
        self.force_multioutput = force_multioutput
        
        # Check for SCP types which aren't compatible
        type_pi = kwargs.get('type_pi', 'kde')
        if type_pi.startswith('scp'):
            raise ValueError(
                "MultiOutputMTS does not support SCP types. "
                "Use regular MTS class for conformal prediction."
            )
        
        # Test multioutput capability
        self._multioutput_supported = self._test_multioutput_capability(obj)
        
        # Initialize parent class with a clone
        super().__init__(clone(obj), **kwargs)
        
        # Additional attributes for multioutput mode
        self._multioutput_model = None
        
    def _test_multioutput_capability(self, obj):
        """Test if the object supports multioutput regression."""
        # Create minimal test data
        X_test = np.random.RandomState(42).randn(5, 3)
        y_test = np.random.RandomState(42).randn(5, 2)  # 2 outputs
        
        try:
            # Try to fit with multioutput
            obj_clone = clone(obj)
            obj_clone.fit(X_test, y_test)
            predictions = obj_clone.predict(X_test)
            
            # Check prediction shape matches
            if predictions.shape != y_test.shape:
                warnings.warn(
                    f"Object {type(obj).__name__} predictions shape {predictions.shape} "
                    f"doesn't match target shape {y_test.shape}. Multioutput may not be fully supported."
                )
                return False
            return True
                
        except Exception as e:
            # Check if error is about multioutput support
            error_msg = str(e).lower()
            multioutput_keywords = ['shape', 'dimension', '2d', 'multioutput', 'array']
            if any(keyword in error_msg for keyword in multioutput_keywords):
                warnings.warn(
                    f"Object {type(obj).__name__} does not support multioutput regression. "
                    f"Will fall back to loop-based fitting. Error: {e}"
                )
                return False
            else:
                # Other error during test, but might still work with real data
                warnings.warn(
                    f"Error during multioutput capability test: {e}. "
                    f"Will attempt multioutput but fall back if needed."
                )
                return True  # Give it a chance
    
    def fit(self, X, xreg=None, **kwargs):
        """Fit MTS model using multioutput regression for efficiency.
        
        Parameters:
        -----------
        X: {array-like}, shape = [n_samples, n_features]
            Training time series, where n_samples is the number
            of samples and n_features is the number of features;
            X must be in increasing order (most recent observations last)
            
        xreg: {array-like}, shape = [n_samples, n_features_xreg]
            Additional (external) regressors to be passed to self.obj
            xreg must be in 'increasing' order (most recent observations last)
            
        **kwargs: additional parameters to be passed to the base object fit method
            
        Returns:
        --------
        self: object
        """
        # Determine dimensions
        try:
            self.init_n_series_ = X.shape[1]
        except (IndexError, AttributeError):
            self.init_n_series_ = 1
        
        # Check if we should use multioutput
        use_multioutput = (
            self._multioutput_supported and 
            self.init_n_series_ > 1 and
            not self.type_pi.startswith('scp') and
            (self.force_multioutput or self.init_n_series_ > 1)
        )
        
        if not use_multioutput:
            # Fall back to parent MTS method
            if self.verbose > 0:
                print("MultiOutputMTS: Falling back to parent MTS (loop-based fitting)")
            return super().fit(X, xreg=xreg, **kwargs)
        
        # Proceed with multioutput fitting
        return self._fit_multioutput(X, xreg=xreg, **kwargs)
    
    def _fit_multioutput(self, X, xreg=None, **kwargs):
        """Internal method for multioutput fitting."""
        # Set up basic attributes
        self.input_dates = None
        self.df_ = None
        
        # Prepare data (similar to parent but optimized)
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
                X = deepcopy(mo.convert_df_to_numeric(X))
            else:
                X = deepcopy(mo.cbind(mo.convert_df_to_numeric(X), xreg))
                self.xreg_ = xreg
            if X_index is not None:
                X.index = X_index
            self.series_names = X.columns.tolist()
        
        # Store as DataFrame for date handling
        if self.df_ is None:
            self.df_ = X.copy()
            X = X.values
        else:
            # Append to existing data
            self.df_ = pd.concat([self.df_, X], axis=0)
            X = self.df_.values
        
        # Get dimensions
        n, p = X.shape
        self.n_obs_ = n
        self.n_series = p
        
        # Create training inputs (most recent first)
        if p > 1:
            mts_input = ts.create_train_inputs(X[::-1], self.lags)
        else:
            mts_input = ts.create_train_inputs(X.reshape(-1, 1)[::-1], self.lags)
        
        self.y_ = mts_input[0]  # Targets
        self.X_ = mts_input[1]  # Features (lags)
        
        # Cook training set
        dummy_y, scaled_Z = self.cook_training_set(
            y=np.ones(self.y_.shape[0]), 
            X=self.X_
        )
        self.scaled_Z_ = scaled_Z
        
        if self.verbose > 0:
            print(f"MultiOutputMTS: Fitting {p} series using multioutput regression...")
        
        # Center the targets
        y_means = np.mean(self.y_, axis=0, keepdims=True)
        self.y_means_ = y_means.flatten()  # Store as array
        
        centered_y = self.y_ - y_means
        self.centered_y_is_ = centered_y
        
        # Fit multioutput model
        try:
            # Handle quantile regression differently
            if self.type_pi == "quantile":
                self.obj.fit(X=scaled_Z, y=centered_y, **kwargs)
            else:
                self.obj.fit(X=scaled_Z, y=centered_y)
            
            # FIX: Store reference to the fitted model (not a clone!)
            self._multioutput_model = self.obj
            self.fit_objs_['multioutput'] = self.obj
            
            # Calculate residuals
            predictions = self.obj.predict(scaled_Z)
            self.residuals_ = centered_y - predictions
            
            # Verify residuals shape
            if self.residuals_.shape != centered_y.shape:
                raise ValueError(
                    f"Residuals shape mismatch: {self.residuals_.shape} != {centered_y.shape}"
                )
                
        except Exception as e:
            warnings.warn(
                f"Multioutput fitting failed: {str(e)}. "
                f"Reverting to parent MTS method."
            )
            # Clear state and use parent
            self.fit_objs_.clear()
            self._multioutput_model = None
            return super().fit(X if isinstance(X, np.ndarray) else self.df_, 
                            xreg=xreg, **kwargs)
        
        # Post-processing for prediction intervals
        if self.type_pi == "gaussian":
            self.gaussian_preds_std_ = np.std(self.residuals_, axis=0)
        
        if self.type_pi.startswith("scp2"):
            data_mean = np.mean(self.residuals_, axis=0)
            self.residuals_std_dev_ = np.std(self.residuals_, axis=0)
            self.residuals_ = (
                self.residuals_ - data_mean[np.newaxis, :]
            ) / self.residuals_std_dev_[np.newaxis, :]
        
        # Kernel density estimation if requested
        if self.replications is not None and "kde" in self.type_pi:
            if self.verbose > 0:
                print(f"Fitting KDE for residual simulation...")
            from sklearn.neighbors import KernelDensity
            from sklearn.model_selection import GridSearchCV
            
            kernel_bandwidths = {"bandwidth": np.logspace(-6, 6, 150)}
            grid = GridSearchCV(
                KernelDensity(kernel=self.kernel, **kwargs),
                param_grid=kernel_bandwidths,
            )
            grid.fit(self.residuals_)
            self.kde_ = grid.best_estimator_
        
        if self.verbose > 0:
            print("MultiOutputMTS: Fitting completed successfully")
        
        return self
    
    def predict(self, h=5, level=95, quantiles=None, **kwargs):
        """Forecast using multioutput predictions.
        
        Parameters:
        -----------
        h: int, default=5
            Forecast horizon (number of steps ahead)
            
        level: int or list, default=95
            Confidence level for prediction intervals (e.g., 95 for 95%)
            
        quantiles: array-like, optional
            Specific quantiles to predict (between 0 and 1)
            
        **kwargs: additional parameters for prediction
            
        Returns:
        --------
        If quantiles provided: DataFrame with quantile predictions
        If level provided: NamedTuple with mean, lower, upper (and sims if applicable)
        Otherwise: DataFrame with point forecasts
        """
        # Check if we have a multioutput model
        if self._multioutput_model is None or 'multioutput' not in self.fit_objs_:
            if self.verbose > 0:
                print("MultiOutputMTS: Using parent MTS predict (multioutput not available)")
            return super().predict(h=h, level=level, quantiles=quantiles, **kwargs)
        
        # Handle quantile prediction
        if quantiles is not None:
            return self._predict_quantiles(h=h, quantiles=quantiles, **kwargs)
        
        # Handle multiple levels
        if isinstance(level, (list, np.ndarray)):
            return super().predict(h=h, level=level, quantiles=quantiles, **kwargs)
        
        # Single level prediction with multioutput
        self.output_dates_, _ = ts.compute_output_dates(self.df_, h)
        self.level_ = level
        self.alpha_ = 100 - level
        
        # Initialize recursive prediction
        current_state = self.y_.copy()
        predictions = []
        
        for step in range(h):
            # Create input from current state
            new_obs = ts.reformat_response(current_state, self.lags)
            new_X = new_obs.reshape(1, -1)
            cooked_new_X = self.cook_test_set(new_X, **kwargs)
            
            # Multioutput prediction
            predicted_centered = self._multioutput_model.predict(cooked_new_X)
            predicted_centered = predicted_centered.squeeze()
            
            # Add back means
            preds = self.y_means_ + predicted_centered
            
            # Handle external regressors if present
            if self.xreg_ is not None and "xreg" in kwargs:
                next_xreg = kwargs["xreg"].iloc[step:step+1].values.flatten()
                full_row = np.concatenate([preds, next_xreg])
            else:
                full_row = preds
            
            # Update state for next iteration
            new_row = np.zeros((1, current_state.shape[1]))
            new_row[0, :full_row.shape[0]] = full_row
            current_state = np.vstack([new_row, current_state[:-1]])
            
            predictions.append(preds)
        
        # Create DataFrame with predictions
        predictions_array = np.array(predictions)[::-1]  # Reverse to chronological
        self.mean_ = pd.DataFrame(
            predictions_array,
            columns=self.series_names[:self.init_n_series_],
            index=self.output_dates_,
        )
        
        # Generate prediction intervals based on type_pi
        if self.type_pi == "gaussian":
            return self._predict_gaussian_intervals()
        elif self.type_pi in ("kde", "bootstrap", "block-bootstrap") or "vine" in self.type_pi:
            return self._predict_with_simulation(h, level)
        elif self.type_pi == "quantile":
            return self._predict_quantile_only()
        else:
            return self.mean_
    
    def _predict_gaussian_intervals(self):
        """Handle Gaussian prediction intervals."""
        from collections import namedtuple
        
        DescribeResult = namedtuple("DescribeResult", ("mean", "lower", "upper"))
        
        # Use standard deviation of residuals
        if hasattr(self, 'preds_std_') and len(self.preds_std_) > 0:
            preds_std = np.asarray(self.preds_std_)
        else:
            preds_std = self.gaussian_preds_std_
        
        pi_multiplier = norm.ppf(1 - self.alpha_ / 200)
        
        self.lower_ = pd.DataFrame(
            self.mean_.values - pi_multiplier * preds_std,
            columns=self.series_names[:self.init_n_series_],
            index=self.output_dates_,
        )
        
        self.upper_ = pd.DataFrame(
            self.mean_.values + pi_multiplier * preds_std,
            columns=self.series_names[:self.init_n_series_],
            index=self.output_dates_,
        )
        
        return DescribeResult(self.mean_, self.lower_, self.upper_)
    
    def _predict_with_simulation(self, h, level):
        """Handle prediction with simulation-based intervals."""
        from collections import namedtuple
        
        # Generate simulations if needed
        if self.replications is not None and self.residuals_sims_ is None:
            self._generate_residual_simulations(h)
        
        if self.replications is None:
            return self.mean_
        
        # Aggregate simulations
        mean_forecasts = []
        lower_bounds = []
        upper_bounds = []
        
        if "scp2" in self.type_pi:
            self.sims_ = tuple(
                self.mean_ + self.residuals_sims_[i] * self.residuals_std_dev_[np.newaxis, :]
                for i in range(self.replications)
            )
        else:
            self.sims_ = tuple(
                self.mean_ + self.residuals_sims_[i]
                for i in range(self.replications)
            )
        
        DescribeResult = namedtuple("DescribeResult", ("mean", "sims", "lower", "upper"))
        
        for series_idx in range(self.init_n_series_):
            sims_series = getsims(self.sims_, series_idx)
            
            if self.agg == "mean":
                mean_forecasts.append(np.mean(sims_series, axis=1))
            else:
                mean_forecasts.append(np.median(sims_series, axis=1))
            
            lower_bounds.append(np.quantile(sims_series, q=self.alpha_ / 200, axis=1))
            upper_bounds.append(np.quantile(sims_series, q=1 - self.alpha_ / 200, axis=1))
        
        self.mean_ = pd.DataFrame(
            np.array(mean_forecasts).T,
            columns=self.series_names[:self.init_n_series_],
            index=self.output_dates_,
        )
        
        self.lower_ = pd.DataFrame(
            np.array(lower_bounds).T,
            columns=self.series_names[:self.init_n_series_],
            index=self.output_dates_,
        )
        
        self.upper_ = pd.DataFrame(
            np.array(upper_bounds).T,
            columns=self.series_names[:self.init_n_series_],
            index=self.output_dates_,
        )
        
        return DescribeResult(self.mean_, self.sims_, self.lower_, self.upper_)
    
    def _generate_residual_simulations(self, h):
        """Generate residual simulations based on type_pi."""
        target_cols = self.series_names[:self.init_n_series_]
        
        if self.kde_ is not None:
            # KDE-based simulation
            self.residuals_sims_ = tuple(
                pd.DataFrame(
                    self.kde_.sample(n_samples=h, random_state=self.seed + 100 * i),
                    columns=target_cols,
                    index=self.output_dates_,
                )
                for i in range(self.replications)
            )
        elif "bootstrap" in self.type_pi:
            # Bootstrap-based simulation
            block_size = self.block_size if "block" in self.type_pi else None
            self.residuals_sims_ = tuple(
                ts.bootstrap(
                    self.residuals_,
                    h=h,
                    block_size=block_size,
                    seed=self.seed + 100 * i,
                )
                for i in range(self.replications)
            )
        elif "vine" in self.type_pi:
            # Vine copula simulation
            self.residuals_sims_ = tuple(
                vinecopula_sample(
                    x=self.residuals_,
                    n_samples=h,
                    method=self.type_pi,
                    random_state=self.seed + 100 * i,
                )
                for i in range(self.replications)
            )
    
    def get_multioutput_info(self):
        """Get information about multioutput support and status.
        
        Returns:
            dict: Information about multioutput capabilities and current status.
        """
        return {
            'multioutput_supported': self._multioutput_supported,
            'multioutput_used': self._multioutput_model is not None,
            'force_multioutput': self.force_multioutput,
            'n_series': self.init_n_series_,
            'base_object_type': type(self.obj).__name__,
            'type_pi': self.type_pi,
            'scp_compatible': not self.type_pi.startswith('scp')
        }
    
    def _predict_quantiles(self, h, quantiles, **kwargs):
        """Predict arbitrary quantiles from simulated paths."""
        # Ensure we have simulations
        if self.sims_ is None:
            # Trigger simulation generation
            _ = self.predict(h=h, level=95, **kwargs)
        
        # Stack simulations
        sims_array = np.stack([sim.values for sim in self.sims_], axis=0)
        
        # Compute quantiles
        q_values = np.quantile(sims_array, quantiles, axis=0)
        
        # Create result DataFrame
        result_dict = {}
        for i, q in enumerate(quantiles):
            q_label = f"{int(q * 100):02d}" if (q * 100).is_integer() else f"{q:.3f}".replace(".", "_")
            for series_id in range(self.init_n_series_):
                series_name = self.series_names[series_id]
                col_name = f"quantile_{q_label}_{series_name}"
                result_dict[col_name] = q_values[i, :, series_id]
        
        return pd.DataFrame(result_dict, index=self.output_dates_)
    
    def _predict_quantile_only(self):
        """Handle quantile-only prediction."""
        from collections import namedtuple
        DescribeResult = namedtuple("DescribeResult", ("mean",))
        return DescribeResult(self.mean_)