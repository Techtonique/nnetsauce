import numpy as np
import pandas as pd
from collections import namedtuple
from scipy.stats import norm
from sklearn.linear_model import Ridge, RidgeCV
import matplotlib.pyplot as plt


class MLARCH:
    """Machine Learning-agnostic ARCH for nearly-stationary time series (e.g., returns)

    Parameters
    ----------
    model_mean : object
        Model for mean component
    model_sigma : object
        Model for volatility component (sklearn regressor)
    model_residuals : object
        Model for standardized residuals
    lags_vol : int, default=10
        Number of lags for squared residuals in volatility model
    """

    def __init__(self, model_mean, model_sigma, model_residuals, lags_vol=10):
        self.model_mean = model_mean
        self.model_sigma = model_sigma
        self.model_residuals = model_residuals
        self.lags_vol = lags_vol

    def _create_lags(self, y, lags):
        """Create lagged feature matrix"""
        n = len(y)
        if n <= lags:
            raise ValueError(f"Series length {n} must be > lags {lags}")
        X = np.zeros((n - lags, lags))
        for i in range(lags):
            X[:, i] = y[i: (n - lags + i)]
        return X

    def fit(self, y, **kwargs):
        """Fit the MLARCH model

        Parameters
        ----------
        y : array-like
            Target time series (should be stationary, e.g., returns)

        Returns
        -------
        self
        """
        # Format input
        if isinstance(y, (pd.Series, pd.DataFrame)):
            y = y.values
        y = y.ravel()

        if len(y) < self.lags_vol + 20:
            raise ValueError(f"Need at least {self.lags_vol + 20} observations")

        # Step 1: Fit mean model
        self.model_mean.fit(y.reshape(-1, 1))
        mean_residuals = self.model_mean.residuals_.ravel()

        # Step 2: Fit ARCH volatility model on lagged squared residuals
        resid_squared = mean_residuals**2
        X_vol = self._create_lags(resid_squared, self.lags_vol)
        y_vol = np.log(resid_squared[self.lags_vol:] + 1e-8)

        self.model_sigma.fit(X_vol, y_vol)

        # Get fitted volatility
        fitted_log_sigma = self.model_sigma.predict(X_vol)
        fitted_sigma = np.exp(fitted_log_sigma)

        # Step 3: Compute standardized residuals with proper scaling
        standardized_residuals = mean_residuals[self.lags_vol:] / np.sqrt(
            fitted_sigma
        )

        # Enforce zero mean and unit variance
        self.z_mean_ = np.mean(standardized_residuals)
        self.z_std_ = np.std(standardized_residuals)
        standardized_residuals = (
            standardized_residuals - self.z_mean_
        ) / self.z_std_

        # Step 4: Fit residuals model
        self.model_residuals.fit(standardized_residuals.reshape(-1, 1))

        # Store for prediction
        self.last_residuals_squared_ = resid_squared[-self.lags_vol:]

        # Store diagnostics
        self.fitted_volatility_mean_ = np.mean(np.sqrt(fitted_sigma))
        self.fitted_volatility_std_ = np.std(np.sqrt(fitted_sigma))

        return self

    def predict(self, h=5, level=95, return_sims=False):
        """Predict future values

        Parameters
        ----------
        h : int
            Forecast horizon
        level : int
            Confidence level for prediction intervals
        return_sims : bool
            If True, return full simulation paths

        Returns
        -------
        DescribeResult
            Named tuple with mean, sims, lower, upper
        """
        DescribeResult = namedtuple(
            "DescribeResult", ("mean", "sims", "lower", "upper")
        )

        # Get mean forecast
        mean_forecast = self.model_mean.predict(h=h).values.ravel()

        # Recursive ARCH volatility forecasting
        sigma_forecast = np.zeros(h)
        current_lags = self.last_residuals_squared_.copy()

        for i in range(h):
            X_t = current_lags.reshape(1, -1)
            log_sigma_t = self.model_sigma.predict(X_t)[0]
            sigma_forecast[i] = np.exp(log_sigma_t)
            # Update lags with predicted variance
            current_lags = np.append(current_lags[1:], sigma_forecast[i])

        # Predict standardized residuals and rescale
        z_forecast_normalized = self.model_residuals.predict(h=h).values.ravel()
        z_forecast = z_forecast_normalized * self.z_std_ + self.z_mean_

        # Combine: μ + z × σ
        point_forecast = mean_forecast + z_forecast * np.sqrt(sigma_forecast)

        # Generate prediction intervals
        sims = None
        if return_sims:
            preds_z_for_sims = self.model_residuals.predict(h=h)
            if hasattr(preds_z_for_sims, "sims") and isinstance(
                preds_z_for_sims.sims, pd.DataFrame
            ):
                sims_z_normalized = preds_z_for_sims.sims
                n_sims = sims_z_normalized.shape[1]

                sims = np.zeros((h, n_sims))
                for sim_idx in range(n_sims):
                    # Rescale simulations
                    z_sim = (
                        sims_z_normalized.iloc[:, sim_idx].values * self.z_std_
                        + self.z_mean_
                    )
                    sims[:, sim_idx] = mean_forecast + z_sim * np.sqrt(
                        sigma_forecast
                    )

                alpha = 1 - level / 100
                lower_bound = np.quantile(sims, alpha / 2, axis=1)
                upper_bound = np.quantile(sims, 1 - alpha / 2, axis=1)
            else:
                # Fallback to Gaussian
                z_score = norm.ppf(1 - (1 - level / 100) / 2)
                margin = z_score * np.sqrt(sigma_forecast) * self.z_std_
                lower_bound = point_forecast - margin
                upper_bound = point_forecast + margin
        else:
            # Gaussian intervals with proper scaling
            z_score = norm.ppf(1 - (1 - level / 100) / 2)
            margin = z_score * np.sqrt(sigma_forecast) * self.z_std_
            lower_bound = point_forecast - margin
            upper_bound = point_forecast + margin

        return DescribeResult(point_forecast, sims, lower_bound, upper_bound)
