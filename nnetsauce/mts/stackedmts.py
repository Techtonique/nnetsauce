import numpy as np
import pandas as pd
from sklearn.base import clone
from copy import deepcopy
from collections import namedtuple
from .mts import MTS 
from ..utils import matrixops as mo
from ..utils import misc as mx

class MTSStacker(MTS):
    """
    Sequential stacking for time series with unified strategy.

    Core Strategy:
    1. Split data: half1 (base models) | half2 (meta-model)
    2. Train base models on half1, predict half2
    3. Create augmented dataset: [original_series | base_pred_1 | base_pred_2 | ...]
    Stack as additional time series, extract target series
    4. Train meta-MTS on half2 with augmented data (via multivariate or xreg)
    5. Retrain base models on half2 for temporal alignment
    6. At prediction: base models forecast → augment → meta-model predicts
    """
    def __init__(
        self,
        base_models,
        meta_model,
        split_ratio=0.5,
    ):
        """
        Parameters
        ----------
        base_models : list of sklearn-compatible models
            Base models (e.g., Ridge, Lasso, RandomForest)
        meta_model : nnetsauce.MTS instance
            MTS with type_pi='scp2-kde' or similar
        split_ratio : float
            Proportion for half1 (default: 0.5)
        """
        self.base_models = base_models
        self.meta_model = meta_model
        self.split_ratio = split_ratio
        self.fitted_base_models_ = []
        self.split_idx_ = None
        self.mean_ = None
        self.lower_ = None
        self.upper_ = None
        self.output_dates_ = None

    def fit(self, X, xreg=None, **kwargs):
        """
        Fit MTSStacker using sequential stacking strategy.
        
        Parameters
        ----------
        X : array-like or DataFrame, shape (n_samples, n_features)
            Training time series (most recent observations last)
        xreg : array-like, optional
            External regressors
        **kwargs : dict
            Additional parameters for base and meta models
        
        Returns
        -------
        self : object
        """
        # 1. Store attributes and convert to DataFrame if needed
        if isinstance(X, pd.DataFrame):
            self.df_ = X.copy()
            X_array = X.values
            self.series_names = X.columns.tolist()
        else:
            X_array = np.asarray(X)
            self.df_ = pd.DataFrame(X_array)
            self.series_names = [f"series{i}" for i in range(X_array.shape[1])]
        
        n_samples = X_array.shape[0]
        self.n_series_ = X_array.shape[1] if X_array.ndim > 1 else 1
        
        # 2. Split data into half1 and half2
        split_idx = int(n_samples * self.split_ratio)
        self.split_idx_ = split_idx
        half1 = X_array[:split_idx]
        half2 = X_array[split_idx:]
        
        # 3. Train base models on half1 and predict half2
        base_preds = []
        self.fitted_base_models_ = []
        
        for base_model in self.base_models:
            # Wrap in MTS with same config as meta_model
            base_mts = MTS(
                obj=clone(base_model),
                lags=self.meta_model.lags,
                n_hidden_features=self.meta_model.n_hidden_features
            )
            base_mts.fit(half1)            
            # Predict half2
            pred = base_mts.predict(h=len(half2))
            base_preds.append(pred.values if isinstance(pred, pd.DataFrame) else pred)
            self.fitted_base_models_.append(base_mts)
                    
        # 4. Create augmented dataset: [original | base_pred_1 | base_pred_2 | ...]
        base_preds_array = np.hstack(base_preds)  # shape: (len(half2), n_series * n_base_models)
        
        if isinstance(X, pd.DataFrame):
            half2_df = pd.DataFrame(half2, index=self.df_.index[split_idx:], columns=self.series_names)
            base_preds_df = pd.DataFrame(base_preds_array, 
                                       index=self.df_.index[split_idx:],
                                       columns=[f"base_pred_{i}" for i in range(base_preds_array.shape[1])])
            augmented = pd.concat([half2_df, base_preds_df], axis=1)
        else:
            augmented = np.hstack([half2, base_preds_array])
        
        # 5. Train meta-model on augmented half2
        self.meta_model.fit(augmented, xreg=xreg, **kwargs)

        # Store meta-model attributes
        self.output_dates_ = self.meta_model.output_dates_
        self.fit_objs_ = self.meta_model.fit_objs_
        self.y_ = self.meta_model.y_
        self.X_ = self.meta_model.X_
        self.xreg_ = self.meta_model.xreg_
        self.y_means_ = self.meta_model.y_means_
        self.residuals_ = self.meta_model.residuals_
        
        return self

    def predict(self, h=5, level=95, **kwargs):
        """
        Forecast h steps ahead using stacked predictions.
        
        Parameters
        ----------
        h : int
            Forecast horizon
        level : int
            Confidence level for prediction intervals
        **kwargs : dict
            Additional parameters for prediction
        
        Returns
        -------
        DescribeResult or DataFrame
            Predictions with optional intervals/simulations
        """
        # Meta-model predicts all series (original + base predictions)
        result = self.meta_model.predict(h=h, level=level, **kwargs)

        # Store output dates from meta-model
        self.output_dates_ = self.meta_model.output_dates_

        def create_result_with_correct_dates(meta_result, n_series, output_dates, series_names):
            """Helper function to extract original series with correct date alignment"""
            
            def extract_original_series(x):
                """Extract only the original series from meta-model results"""
                if isinstance(x, pd.DataFrame):
                    # Take first n_series columns and ensure correct dates
                    sliced = x.iloc[:, :n_series].copy()
                    sliced.index = output_dates
                    sliced.columns = series_names[:n_series]
                    return sliced
                elif isinstance(x, np.ndarray):
                    # For arrays, just slice columns
                    return x[:, :n_series]
                elif isinstance(x, tuple):
                    # Handle tuple of DataFrames/arrays (e.g., sims)
                    if all(isinstance(item, pd.DataFrame) for item in x):
                        # For sims, process each DataFrame individually
                        return tuple(
                            extract_original_series(item) for item in x
                        )
                    else:
                        return tuple(extract_original_series(item) for item in x)
                else:
                    return x
            
            return mx.tuple_map(meta_result, extract_original_series)

        # Handle different return types from meta-model
        if isinstance(result, pd.DataFrame):
            # Simple DataFrame case
            sliced = result.iloc[:, :self.n_series_].copy()
            sliced.index = self.output_dates_
            sliced.columns = self.series_names[:self.n_series_]
            return sliced
            
        elif isinstance(result, np.ndarray):
            # Simple array case
            return result[:, :self.n_series_]
            
        else:
            # Namedtuple case (with or without sims)
            processed_result = create_result_with_correct_dates(
                result, self.n_series_, self.output_dates_, self.series_names
            )
            
            # Store the results for plotting
            if hasattr(self.meta_model, 'sims_') and self.meta_model.sims_ is not None:
                DescribeResult = namedtuple("DescribeResult", ("mean", "sims", "lower", "upper"))
                if len(processed_result) == 4:
                    self.mean_, self.sims_, self.lower_, self.upper_ = processed_result
                    return DescribeResult(self.mean_, self.sims_, self.lower_, self.upper_)
                else:
                    # Handle case where we don't have exactly 4 elements
                    self.mean_, self.sims_, self.lower_, self.upper_ = processed_result[0], processed_result[1], processed_result[2], processed_result[3]
                    return DescribeResult(self.mean_, self.sims_, self.lower_, self.upper_)
            else:
                DescribeResult = namedtuple("DescribeResult", ("mean", "lower", "upper"))
                if len(processed_result) == 3:
                    self.mean_, self.lower_, self.upper_ = processed_result
                    return DescribeResult(self.mean_, self.lower_, self.upper_)
                else:
                    # Handle case where we don't have exactly 3 elements
                    self.mean_, self.lower_, self.upper_ = processed_result[0], processed_result[1], processed_result[2]
                    return DescribeResult(self.mean_, self.lower_, self.upper_)

    def plot(self, series=None, **kwargs):
        """
        Plot the time series.
        
        Parameters
        ----------
        series : str, optional
            Name of the series to plot
        **kwargs : dict
            Additional parameters for plotting
        """
        # Ensure we have predictions
        if self.mean_ is None or self.lower_ is None or self.upper_ is None:
            raise ValueError("Model forecasting must be obtained first (call predict)")
        
        # Convert series name to index if needed
        if isinstance(series, str):
            if series in self.series_names:
                series_idx = self.series_names.index(series)
            else:
                raise ValueError(f"Series '{series}' doesn't exist in the input dataset")
        else:
            series_idx = series if series is not None else 0
        
        # Check bounds
        if series_idx < 0 or series_idx >= self.n_series_:
            raise ValueError(f"Series index {series_idx} is out of bounds (0 to {self.n_series_ - 1})")
        
        # Prepare data for plotting - convert all dates to pandas Timestamp for consistency
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        
        # Get historical data
        historical_data = self.df_.iloc[:, series_idx]
        forecast_data = self.mean_.iloc[:, series_idx]
        lower_data = self.lower_.iloc[:, series_idx]
        upper_data = self.upper_.iloc[:, series_idx]
        
        # Convert indices to consistent datetime format
        if hasattr(self.df_, 'index') and hasattr(self.mean_, 'index'):
            # Convert all indices to pandas DatetimeIndex for consistency
            hist_index = pd.to_datetime(self.df_.index)
            forecast_index = pd.to_datetime(self.mean_.index)
            
            # Combine for full timeline
            full_index = hist_index.union(forecast_index)
            
            # Create figure and axis
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Plot historical data
            ax.plot(hist_index, historical_data, "-", label="Historical", color='blue', linewidth=1.5)
            
            # Plot forecast data
            ax.plot(forecast_index, forecast_data, "-", label="Forecast", color='red', linewidth=1.5)
            
            # Plot prediction intervals
            ax.fill_between(forecast_index, lower_data, upper_data, alpha=0.3, color='red', label="Prediction Interval")
            
            # Add vertical line at the split point
            if hasattr(self, 'split_idx_') and self.split_idx_ is not None:
                split_date = hist_index[-1]  # Last historical date
                ax.axvline(x=split_date, color='gray', linestyle='--', alpha=0.7, label='Train/Test Split')
            
            # Format x-axis for dates
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax.xaxis.set_major_locator(mdates.YearLocator())
            plt.xticks(rotation=45)
            
        else:
            # Fallback to numeric indices if no dates available
            n_points_train = self.df_.shape[0]
            n_points_forecast = self.mean_.shape[0]
            
            x_hist = list(range(n_points_train))
            x_forecast = list(range(n_points_train, n_points_train + n_points_forecast))
            x_full = x_hist + x_forecast
            
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Plot historical data
            ax.plot(x_hist, historical_data, "-", label="Historical", color='blue', linewidth=1.5)
            
            # Plot forecast data
            ax.plot(x_forecast, forecast_data, "-", label="Forecast", color='red', linewidth=1.5)
            
            # Plot prediction intervals
            ax.fill_between(x_forecast, lower_data, upper_data, alpha=0.3, color='red', label="Prediction Interval")
            
            # Add vertical line at the split point
            if hasattr(self, 'split_idx_') and self.split_idx_ is not None:
                ax.axvline(x=n_points_train - 0.5, color='gray', linestyle='--', alpha=0.7, label='Train/Test Split')
        
        # Set title and labels
        series_name = self.series_names[series_idx] if series_idx < len(self.series_names) else f"Series {series_idx}"
        plt.title(f"Forecast for {series_name}", fontsize=14, fontweight='bold')
        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Adjust layout and show
        plt.tight_layout()
        plt.show()