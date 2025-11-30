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
            
            # Determine the type of result and return appropriate namedtuple
            if hasattr(self.meta_model, 'sims_') and self.meta_model.sims_ is not None:
                DescribeResult = namedtuple("DescribeResult", ("mean", "sims", "lower", "upper"))
                if len(processed_result) == 4:
                    return DescribeResult(*processed_result)
                else:
                    # Handle case where we don't have exactly 4 elements
                    return DescribeResult(processed_result[0], processed_result[1], 
                                        processed_result[2], processed_result[3])
            else:
                DescribeResult = namedtuple("DescribeResult", ("mean", "lower", "upper"))
                if len(processed_result) == 3:
                    return DescribeResult(*processed_result)
                else:
                    # Handle case where we don't have exactly 3 elements
                    return DescribeResult(processed_result[0], processed_result[1], processed_result[2])

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
        # First, ensure we have predictions by calling predict if needed
        if not hasattr(self, 'mean_') or self.mean_ is None:
            # Get predictions to populate mean_, lower_, upper_
            _ = self.predict(h=10, level=95)  # Use default h=10 if not specified in kwargs
        
        # Now use the parent MTS class plot method with our own attributes
        # We need to temporarily set the required attributes for plotting
        temp_attrs = {}
        
        # Store original attributes
        original_attrs = {}
        for attr in ['mean_', 'lower_', 'upper_', 'sims_', 'output_dates_', 'series_names', 'n_series', 'init_n_series_']:
            if hasattr(self, attr):
                original_attrs[attr] = getattr(self, attr)
        
        try:
            # Use the parent class plot method directly
            super().plot(series=series, **kwargs)
            
        except AssertionError as e:
            if "doesn't exist in the input dataset" in str(e):
                # Handle the case where series name doesn't match
                if series is not None and series in self.series_names:
                    # The series exists in our names, but the parent class might be using different names
                    # Let's map the series name to an index and use that
                    series_idx = self.series_names.index(series)
                    super().plot(series=series_idx, **kwargs)
                else:
                    # Re-raise the error if we can't handle it
                    raise
            else:
                raise
        finally:
            # Restore original attributes
            for attr, value in original_attrs.items():
                setattr(self, attr, value)

    # Override the attributes that the parent MTS plot method expects
    @property
    def n_series(self):
        return self.n_series_
    
    @property 
    def init_n_series_(self):
        return self.n_series_