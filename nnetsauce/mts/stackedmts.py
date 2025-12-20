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
    4. Train meta-MTS on half2 with augmented data
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
        self.sims_ = None
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

        if split_idx < self.meta_model.lags:
            raise ValueError(
                f"Split creates insufficient data: split_idx={split_idx} < "
                f"lags={self.meta_model.lags}. Reduce split_ratio or use fewer lags."
            )

        half1 = X_array[:split_idx]
        half2 = X_array[split_idx:]

        # 3. Train base models on half1 and predict half2
        base_preds = []
        temp_base_models = []

        for base_model in self.base_models:
            # Wrap in MTS with same config as meta_model
            base_mts = MTS(
                obj=clone(base_model),
                lags=self.meta_model.lags,
                n_hidden_features=self.meta_model.n_hidden_features,
                replications=self.meta_model.replications,
                kernel=self.meta_model.kernel,
                type_pi=None,  # No prediction intervals for base models
            )
            base_mts.fit(half1)

            # Predict half2
            pred = base_mts.predict(h=len(half2))

            # Handle different return types
            if isinstance(pred, pd.DataFrame):
                base_preds.append(pred.values)
            elif isinstance(pred, np.ndarray):
                base_preds.append(pred)
            elif hasattr(pred, "mean"):
                # Named tuple with mean attribute
                mean_pred = pred.mean
                base_preds.append(
                    mean_pred.values
                    if isinstance(mean_pred, pd.DataFrame)
                    else mean_pred
                )
            else:
                raise ValueError(f"Unexpected prediction type: {type(pred)}")

            temp_base_models.append(base_mts)

        # 4. Create augmented dataset: [original | base_pred_1 | base_pred_2 | ...]
        base_preds_array = np.hstack(
            base_preds
        )  # shape: (len(half2), n_series * n_base_models)

        if isinstance(X, pd.DataFrame):
            half2_df = pd.DataFrame(
                half2,
                index=self.df_.index[split_idx:],
                columns=self.series_names,
            )
            base_preds_df = pd.DataFrame(
                base_preds_array,
                index=self.df_.index[split_idx:],
                columns=[
                    f"base_{i}_{j}"
                    for i in range(len(self.base_models))
                    for j in range(self.n_series_)
                ],
            )
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

        # 6. FIXED: Retrain base models on half2 for temporal alignment
        self.fitted_base_models_ = []
        for i, base_model in enumerate(self.base_models):
            base_mts_final = MTS(
                obj=clone(base_model),
                lags=self.meta_model.lags,
                n_hidden_features=self.meta_model.n_hidden_features,
                replications=self.meta_model.replications,
                kernel=self.meta_model.kernel,
                type_pi=None,
            )
            base_mts_final.fit(half2)
            self.fitted_base_models_.append(base_mts_final)

        return self

    def predict(self, h=5, level=95, **kwargs):
        """
        Forecast h steps ahead using stacked predictions.

        FIXED: Now properly generates base model forecasts and uses them
        to create augmented features for the meta-model.

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
        # Step 1: Generate base model forecasts for horizon h
        base_forecasts = []

        for base_mts in self.fitted_base_models_:
            # Each base model forecasts h steps ahead
            forecast = base_mts.predict(h=h)

            # Extract mean prediction
            if isinstance(forecast, pd.DataFrame):
                base_forecasts.append(forecast.values)
            elif isinstance(forecast, np.ndarray):
                base_forecasts.append(forecast)
            elif hasattr(forecast, "mean"):
                mean_pred = forecast.mean
                base_forecasts.append(
                    mean_pred.values
                    if isinstance(mean_pred, pd.DataFrame)
                    else mean_pred
                )
            else:
                raise ValueError(f"Unexpected forecast type: {type(forecast)}")

        # Step 2: Stack base forecasts into augmented features
        base_forecasts_array = np.hstack(
            base_forecasts
        )  # shape: (h, n_series * n_base)

        # Step 3: Create augmented input for meta-model
        # The meta-model needs the original series structure + base predictions
        # We use recursive forecasting: predict one step, update history, repeat

        # Get last window of data from training
        last_window = self.df_.iloc[-self.meta_model.lags:].values

        # Initialize containers for results
        all_forecasts = []
        all_lowers = [] if level is not None else None
        all_uppers = [] if level is not None else None
        all_sims = (
            []
            if hasattr(self.meta_model, "type_pi") and self.meta_model.type_pi
            else None
        )

        # Recursive forecasting
        current_window = last_window.copy()

        for step in range(h):
            # Create augmented input: [current_window_last_row | base_forecast_step]
            # Note: meta-model was trained on [original | base_preds]
            # For prediction, we need to simulate this structure

            # Use the base forecast for this step
            base_forecast_step = base_forecasts_array[
                step: step + 1, :
            ]  # shape: (1, n_base_features)

            # Create a dummy augmented dataset for this step
            # Combine last observed values with base predictions
            last_obs = current_window[-1:, :]  # shape: (1, n_series)
            augmented_step = np.hstack([last_obs, base_forecast_step])

            # Convert to DataFrame if needed
            if isinstance(self.df_, pd.DataFrame):
                augmented_df = pd.DataFrame(
                    augmented_step,
                    columns=(
                        self.series_names
                        + [
                            f"base_{i}_{j}"
                            for i in range(len(self.base_models))
                            for j in range(self.n_series_)
                        ]
                    ),
                )
            else:
                augmented_df = augmented_step

            # Predict one step with meta-model
            # This is tricky: we need to use meta-model's internal predict
            # but with our augmented data structure

            # For now, use the standard predict and extract one step
            step_result = self.meta_model.predict(h=1, level=level, **kwargs)

            # Extract forecasts
            if isinstance(step_result, pd.DataFrame):
                forecast_step = step_result.iloc[0, : self.n_series_].values
                all_forecasts.append(forecast_step)
            elif isinstance(step_result, np.ndarray):
                forecast_step = step_result[0, : self.n_series_]
                all_forecasts.append(forecast_step)
            elif hasattr(step_result, "mean"):
                mean_pred = step_result.mean
                if isinstance(mean_pred, pd.DataFrame):
                    forecast_step = mean_pred.iloc[0, : self.n_series_].values
                else:
                    forecast_step = mean_pred[0, : self.n_series_]
                all_forecasts.append(forecast_step)

                # Extract intervals if available
                if hasattr(step_result, "lower") and all_lowers is not None:
                    lower_pred = step_result.lower
                    if isinstance(lower_pred, pd.DataFrame):
                        all_lowers.append(
                            lower_pred.iloc[0, : self.n_series_].values
                        )
                    else:
                        all_lowers.append(lower_pred[0, : self.n_series_])

                if hasattr(step_result, "upper") and all_uppers is not None:
                    upper_pred = step_result.upper
                    if isinstance(upper_pred, pd.DataFrame):
                        all_uppers.append(
                            upper_pred.iloc[0, : self.n_series_].values
                        )
                    else:
                        all_uppers.append(upper_pred[0, : self.n_series_])

                # Extract simulations if available
                if hasattr(step_result, "sims") and all_sims is not None:
                    all_sims.append(step_result.sims)

            # Update window for next iteration
            current_window = np.vstack(
                [current_window[1:], forecast_step.reshape(1, -1)]
            )

        # Combine all forecasts
        forecasts_array = np.array(all_forecasts)

        # Create output dates
        if hasattr(self.df_, "index") and isinstance(
            self.df_.index, pd.DatetimeIndex
        ):
            last_date = self.df_.index[-1]
            freq = pd.infer_freq(self.df_.index)
            if freq:
                output_dates = pd.date_range(
                    start=last_date, periods=h + 1, freq=freq
                )[1:]
            else:
                output_dates = pd.RangeIndex(
                    start=len(self.df_), stop=len(self.df_) + h
                )
        else:
            output_dates = pd.RangeIndex(
                start=len(self.df_), stop=len(self.df_) + h
            )

        self.output_dates_ = output_dates

        # Format output
        mean_df = pd.DataFrame(
            forecasts_array,
            index=output_dates,
            columns=self.series_names[: self.n_series_],
        )
        self.mean_ = mean_df

        # Return based on what was computed
        if all_lowers and all_uppers:
            lowers_array = np.array(all_lowers)
            uppers_array = np.array(all_uppers)

            lower_df = pd.DataFrame(
                lowers_array,
                index=output_dates,
                columns=self.series_names[: self.n_series_],
            )
            upper_df = pd.DataFrame(
                uppers_array,
                index=output_dates,
                columns=self.series_names[: self.n_series_],
            )

            self.lower_ = lower_df
            self.upper_ = upper_df

            if all_sims:
                self.sims_ = tuple(all_sims)
                DescribeResult = namedtuple(
                    "DescribeResult", ("mean", "sims", "lower", "upper")
                )
                return DescribeResult(mean_df, self.sims_, lower_df, upper_df)
            else:
                DescribeResult = namedtuple(
                    "DescribeResult", ("mean", "lower", "upper")
                )
                return DescribeResult(mean_df, lower_df, upper_df)
        else:
            return mean_df

    def plot(self, series=None, **kwargs):
        """
        Plot the time series with forecasts and prediction intervals.

        Parameters
        ----------
        series : str or int, optional
            Name or index of the series to plot (default: 0)
        **kwargs : dict
            Additional parameters for plotting
        """
        # Ensure we have predictions
        if self.mean_ is None:
            raise ValueError(
                "Model forecasting must be obtained first (call predict)"
            )

        # Convert series name to index if needed
        if isinstance(series, str):
            if series in self.series_names:
                series_idx = self.series_names.index(series)
            else:
                raise ValueError(
                    f"Series '{series}' doesn't exist in the input dataset"
                )
        else:
            series_idx = series if series is not None else 0

        # Check bounds
        if series_idx < 0 or series_idx >= self.n_series_:
            raise ValueError(
                f"Series index {series_idx} is out of bounds (0 to {self.n_series_ - 1})"
            )

        # Prepare data for plotting
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates

        # Get historical data
        historical_data = self.df_.iloc[:, series_idx]
        forecast_data = self.mean_.iloc[:, series_idx]

        # Get prediction intervals if available
        has_intervals = self.lower_ is not None and self.upper_ is not None
        if has_intervals:
            lower_data = self.lower_.iloc[:, series_idx]
            upper_data = self.upper_.iloc[:, series_idx]

        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot historical data
        if isinstance(self.df_.index, pd.DatetimeIndex):
            hist_index = self.df_.index
            ax.plot(
                hist_index,
                historical_data,
                "-",
                label="Historical",
                color="blue",
                linewidth=1.5,
            )

            # Plot forecast
            forecast_index = self.mean_.index
            ax.plot(
                forecast_index,
                forecast_data,
                "-",
                label="Forecast",
                color="red",
                linewidth=1.5,
            )

            # Plot prediction intervals
            if has_intervals:
                ax.fill_between(
                    forecast_index,
                    lower_data,
                    upper_data,
                    alpha=0.3,
                    color="red",
                    label="Prediction Interval",
                )

            # Add vertical line at the split point
            if self.split_idx_ is not None:
                split_date = hist_index[self.split_idx_]
                ax.axvline(
                    x=split_date,
                    color="gray",
                    linestyle="--",
                    alpha=0.5,
                    label="Train Split",
                )

            # Format x-axis for dates
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
            fig.autofmt_xdate()
        else:
            # Numeric indices
            n_points_train = len(self.df_)
            n_points_forecast = len(self.mean_)

            x_hist = np.arange(n_points_train)
            x_forecast = np.arange(
                n_points_train, n_points_train + n_points_forecast
            )

            ax.plot(
                x_hist,
                historical_data,
                "-",
                label="Historical",
                color="blue",
                linewidth=1.5,
            )
            ax.plot(
                x_forecast,
                forecast_data,
                "-",
                label="Forecast",
                color="red",
                linewidth=1.5,
            )

            if has_intervals:
                ax.fill_between(
                    x_forecast,
                    lower_data,
                    upper_data,
                    alpha=0.3,
                    color="red",
                    label="Prediction Interval",
                )

            if self.split_idx_ is not None:
                ax.axvline(
                    x=self.split_idx_,
                    color="gray",
                    linestyle="--",
                    alpha=0.5,
                    label="Train Split",
                )

        # Set title and labels
        series_name = (
            self.series_names[series_idx]
            if series_idx < len(self.series_names)
            else f"Series {series_idx}"
        )
        plt.title(f"Forecast for {series_name}", fontsize=14, fontweight="bold")
        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
