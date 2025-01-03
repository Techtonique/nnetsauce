# Author: @thierrymoudiki
# Date: 2025-01-03

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import scipy 

from collections import namedtuple
from sklearn.base import BaseEstimator, RegressorMixin
from scipy.optimize import differential_evolution
from dataclasses import dataclass
from typing import Dict, Any, List
from sklearn.base import clone
from sklearn.base import BaseEstimator, RegressorMixin
from scipy.optimize import minimize, differential_evolution
from sklearn.ensemble import RandomForestRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import ElasticNet
from sklearn.datasets import load_diabetes, fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from warnings import filterwarnings
filterwarnings('ignore')


@dataclass
class QuantileResult:
    params: Dict[str, Any]
    predictions: np.ndarray
    loss: float
    iterations: int


class QuantileRegressor(BaseEstimator, 
                        RegressorMixin):

    def __init__(self, base_regressor, 
                 level=95, 
                 score="predictions"):
        assert score in ("predictions", 
                         "residuals", 
                         "conformal", 
                         "studentized",
                         "conformal-studentized"),\
         "score must be 'predictions' or 'residuals'"
        self.base_regressor = base_regressor 
        low_risk_level = (1 - level/100)/2       
        self.quantiles = [low_risk_level, 0.5, 1-low_risk_level]
        self.score = score
        self.offset_multipliers_ = None
        self.base_regressor_ = None
        self.scoring_residuals_ = None
        self.student_multiplier_ = None

    def _compute_quantile_loss(self, residuals: np.ndarray, quantile: float) -> float:
        """
        Compute the quantile loss for a given set of residuals and quantile.
        """
        if not 0 < quantile < 1:
            raise ValueError("Quantile should be between 0 and 1.")
        loss = quantile * (residuals >= 0) + (quantile - 1) * (residuals < 0)
        return np.mean(residuals * loss)
    

    def _optimize_multiplier(self, y: np.ndarray, 
                             base_predictions: np.ndarray,
                             prev_predictions: np.ndarray = None,
                             scoring_residuals: np.ndarray = None,
                             quantile: float = 0.5) -> float:
        """
        Optimize the multiplier for a given quantile.
        """
        if not 0 < quantile < 1:
            raise ValueError("Quantile should be between 0 and 1.")
        
        def objective(log_multiplier):
          """
          Objective function for optimization.
          """
          # Convert to positive multiplier using exp
          multiplier = np.exp(log_multiplier[0])   
          if self.score == "predictions":  
            assert base_predictions is not None, "base_predictions must be not None"        
            # Calculate predictions
            if prev_predictions is None:
                # For first quantile, subtract from conditional expectation 
                predictions = base_predictions - multiplier * np.abs(base_predictions)
            else:
                # For other quantiles, add to previous quantile
                offset = multiplier * np.abs(base_predictions)
                predictions = prev_predictions + offset            
          elif self.score in ("residuals", "conformal"):
            assert scoring_residuals is not None, "scoring_residuals must be not None"
            #print("scoring_residuals", scoring_residuals)
            # Calculate predictions
            if prev_predictions is None:
                # For first quantile, subtract from conditional expectation 
                predictions = base_predictions - multiplier * np.std(scoring_residuals)
                #print("predictions", predictions)
            else:
                # For other quantiles, add to previous quantile
                offset = multiplier * np.std(scoring_residuals)
                predictions = prev_predictions + offset            
          elif self.score in ("studentized", "conformal-studentized"):
            assert scoring_residuals is not None, "scoring_residuals must be not None"
            # Calculate predictions
            if prev_predictions is None:
                # For first quantile, subtract from conditional expectation 
                predictions = base_predictions - multiplier * self.student_multiplier_
                #print("predictions", predictions)
            else:
                # For other quantiles, add to previous quantile
                offset = multiplier * self.student_multiplier_
                predictions = prev_predictions + offset
          else:
            raise ValueError("Invalid argument 'score'")

          residuals = y - predictions
          return self._compute_quantile_loss(residuals, quantile)
        
        # Optimize in log space for numerical stability
        #bounds = [(-10, 10)]  # log space bounds
        bounds = [(-100, 100)]  # log space bounds
        result = differential_evolution(objective, 
                                        bounds,
                                        #popsize=15,
                                        #maxiter=100,
                                        #tol=1e-4,
                                        popsize=25,
                                        maxiter=200,
                                        tol=1e-6,
                                        disp=False)
        
        return np.exp(result.x[0])
    

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.base_regressor_ = clone(self.base_regressor)
        if self.score in ("predictions", "residuals"):          
            self.base_regressor_.fit(X, y)
            base_predictions = self.base_regressor_.predict(X)    
            scoring_residuals = y - base_predictions
            self.scoring_residuals_ = scoring_residuals
        elif self.score == "conformal":
            X_train, X_calib, y_train, y_calib = train_test_split(X, y, 
                                                                  test_size=0.5, 
                                                                  random_state=42)
            self.base_regressor_.fit(X_train, y_train)
            scoring_residuals = y_calib - self.base_regressor_.predict(X_calib)  # These are calibration predictions
            self.scoring_residuals_ = scoring_residuals            
            # Update base_predictions to use training predictions for optimization
            self.base_regressor_.fit(X_calib, y_calib)
            base_predictions = self.base_regressor_.predict(X_calib)
        elif self.score in ("studentized", "conformal-studentized"):
            # Calculate student multiplier
            if self.score == "conformal-studentized":
                X_train, X_calib, y_train, y_calib = train_test_split(X, y, 
                                                                      test_size=0.5, 
                                                                      random_state=42)
                self.base_regressor_.fit(X_train, y_train)                
                scoring_residuals = y_calib - self.base_regressor_.predict(X_calib)
                # Calculate studentized multiplier using calibration data
                self.student_multiplier_ = np.std(y_calib, ddof=1)/np.sqrt(len(y_calib))
                self.base_regressor_.fit(X_calib, y_calib)
                base_predictions = self.base_regressor_.predict(X_calib)
            else:  # regular studentized
                self.base_regressor_.fit(X, y)
                base_predictions = self.base_regressor_.predict(X)
                scoring_residuals = y - base_predictions
                self.student_multiplier_ = np.std(y, ddof=1)/np.sqrt(len(y))

        # Initialize storage for multipliers
        self.offset_multipliers_ = []        
        # Keep track of current predictions for each quantile
        current_predictions = None
        
        # Fit each quantile sequentially
        for i, quantile in enumerate(self.quantiles):
            if self.score == "predictions":           
                multiplier = self._optimize_multiplier(
                    y=y, 
                    base_predictions=base_predictions,
                    prev_predictions=current_predictions,
                    quantile=quantile)
                
                self.offset_multipliers_.append(multiplier) 

                # Update current predictions
                if current_predictions is None:
                    # First quantile (lowest)
                    current_predictions = base_predictions - multiplier * np.abs(base_predictions)
                else:
                    # Subsequent quantiles
                    offset = multiplier * np.abs(base_predictions)
                    current_predictions = current_predictions + offset
            elif self.score == "residuals":           
                multiplier = self._optimize_multiplier(
                    y=y, 
                    base_predictions=base_predictions,
                    scoring_residuals=scoring_residuals,
                    prev_predictions=current_predictions,
                    quantile=quantile)
                
                self.offset_multipliers_.append(multiplier) 

                # Update current predictions
                if current_predictions is None:
                    # First quantile (lowest)
                    current_predictions = base_predictions - multiplier * np.std(scoring_residuals)
                else:
                    # Subsequent quantiles
                    offset = multiplier * np.std(scoring_residuals)
                    current_predictions = current_predictions + offset
            elif self.score == "conformal":
                multiplier = self._optimize_multiplier(
                    y=y_calib,
                    base_predictions=base_predictions,
                    scoring_residuals=scoring_residuals,
                    prev_predictions=current_predictions,
                    quantile=quantile)
                
                self.offset_multipliers_.append(multiplier) 

                # Update current predictions
                if current_predictions is None:
                    # First quantile (lowest)
                    current_predictions = base_predictions - multiplier * np.std(scoring_residuals)
                else:
                    # Subsequent quantiles
                    offset = multiplier * np.std(scoring_residuals)
                    current_predictions = current_predictions + offset
            elif self.score in ("studentized", "conformal-studentized"):
                multiplier = self._optimize_multiplier(
                    y=y_calib if self.score == "conformal-studentized" else y,
                    base_predictions=base_predictions,
                    scoring_residuals=scoring_residuals,
                    prev_predictions=current_predictions,
                    quantile=quantile)
                
                self.offset_multipliers_.append(multiplier)

                # Update current predictions
                if current_predictions is None:
                    current_predictions = base_predictions - multiplier * self.student_multiplier_
                else:
                    offset = multiplier * self.student_multiplier_
                    current_predictions = current_predictions + offset

        return self
    
    def predict(self, X):

        if self.base_regressor_ is None or self.offset_multipliers_ is None:
            raise ValueError("Model not fitted yet.")
        
        base_predictions = self.base_regressor_.predict(X)
        all_predictions = []

        if self.score == "predictions": 
        
          # Generate first quantile
          current_predictions = base_predictions - self.offset_multipliers_[0] * np.abs(base_predictions)
          all_predictions.append(current_predictions)
          
          # Generate remaining quantiles
          for multiplier in self.offset_multipliers_[1:]:
              offset = multiplier * np.abs(base_predictions)
              current_predictions = current_predictions + offset
              all_predictions.append(current_predictions)
        
        elif self.score in ("residuals", "conformal"):

          # Generate first quantile
          current_predictions = base_predictions - self.offset_multipliers_[0] * np.std(self.scoring_residuals_)
          all_predictions.append(current_predictions)
          
          # Generate remaining quantiles
          for multiplier in self.offset_multipliers_[1:]:
              offset = multiplier * np.std(self.scoring_residuals_)
              current_predictions = current_predictions + offset
              all_predictions.append(current_predictions)

        elif self.score in ("studentized", "conformal-studentized"):
          # Generate first quantile
          current_predictions = base_predictions - self.offset_multipliers_[0] * self.student_multiplier_
          all_predictions.append(current_predictions)
          
          # Generate remaining quantiles
          for multiplier in self.offset_multipliers_[1:]:
              offset = multiplier * self.student_multiplier_
              current_predictions = current_predictions + offset
              all_predictions.append(current_predictions)
        
        DescribeResult = namedtuple("DecribeResult", ["mean", "lower", "upper", "median"])
        DescribeResult.mean = base_predictions
        DescribeResult.lower = np.asarray(all_predictions[0])
        DescribeResult.median = np.asarray(all_predictions[1])
        DescribeResult.upper = np.asarray(all_predictions[2])        
        return DescribeResult


