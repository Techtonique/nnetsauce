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
        else:
            X_array = np.asarray(X)
            self.df_ = pd.DataFrame(X_array)
        
        n_samples = X_array.shape[0]
        self.n_series_ = X_array.shape[1] if X_array.ndim > 1 else 1
        
        # 2. Split data into half1 and half2
        split_idx = int(n_samples * self.split_ratio)
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
                    
        # 4. Create augmented dataset: [original | base_pred_1 | base_pred_2 | ...]
        base_preds_array = np.hstack(base_preds)  # shape: (len(half2), n_series * n_base_models)
        augmented = np.hstack([half2, base_preds_array])
        
        # 5. Train meta-model on augmented half2
        self.meta_model.fit(augmented, xreg=xreg, **kwargs)
        
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

        if isinstance(result, pd.DataFrame):
            return result.iloc[:, :self.n_series_]
        elif isinstance(result, np.ndarray):
            return result[:, :self.n_series_]
        
        DescribeResult = namedtuple(
                "DescribeResult", ("mean", "lower", "upper")
            )

        # it's a tuple of (mean, lower, upper) or (mean, sims, lower, upper)
        # Extract only the first n_series_ columns (original series predictions)
        def slice_element(x):
            """Slice an element to keep only first n_series_ columns."""
            if isinstance(x, pd.DataFrame):
                return x.iloc[:, :self.n_series_]
            elif isinstance(x, np.ndarray):
                return x[:, :self.n_series_]
            elif isinstance(x, tuple):
                # Handle tuple of DataFrames/arrays (e.g., sims)
                return tuple(slice_element(item) for item in x)
            else:
                # Fallback for other types
                return x
        
        res = mx.tuple_map(result, slice_element)
        return DescribeResult(res[0], res[1], res[2])




            
