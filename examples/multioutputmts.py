"""
Example script demonstrating MultiOutputMTS usage
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor

# Import the class (assuming it's in your Python path)
try:
    import nnetsauce as ns
    print("Successfully imported nnetsauce")
except ImportError:
    print("Creating a dummy class for demonstration")
    # For demonstration, we'll use the class definition above
    # In practice, you would import from nnetsauce


def create_sample_data(n_samples=200, n_series=3, noise=0.1, trend=True):
    """Create correlated multivariate time series."""
    np.random.seed(42)
    
    # Base trend
    if trend:
        base_trend = np.linspace(0, 10, n_samples)
    else:
        base_trend = np.zeros(n_samples)
    
    # Create series with correlations
    series_data = np.zeros((n_samples, n_series))
    
    for i in range(n_series):
        # Each series has: base trend + seasonal pattern + noise + series-specific offset
        seasonal = 2 * np.sin(2 * np.pi * np.arange(n_samples) / 50 + i * np.pi/3)
        noise_component = noise * np.random.randn(n_samples)
        series_data[:, i] = base_trend + seasonal + noise_component + i * 5
    
    # Add some cross-correlation
    for i in range(1, n_series):
        series_data[:, i] += 0.3 * series_data[:, i-1]
    
    # Create DataFrame with dates
    dates = pd.date_range('2020-01-01', periods=n_samples, freq='D')
    df = pd.DataFrame(series_data, 
                      columns=[f'Series_{i}' for i in range(n_series)],
                      index=dates)
    
    return df


def example_1_native_multioutput():
    """Example 1: Using Ridge regression (native multioutput support)."""
    print("\n" + "="*60)
    print("EXAMPLE 1: Ridge Regression (Native Multioutput)")
    print("="*60)
    
    # Create sample data
    df = create_sample_data(n_samples=150, n_series=3)
    print(f"Data shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Create Ridge regressor
    ridge = Ridge(alpha=1.0, random_state=42)
    
    # Create MultiOutputMTS model
    model = ns.MultiOutputMTS(
        obj=ridge,
        lags=15,
        n_hidden_features=15,
        type_pi='kde',
        replications=200,
        kernel='gaussian',
        seed=42,
        verbose=1
    )
    
    # Fit the model
    print("\nFitting model...")
    start_time = time.time()
    model.fit(df)
    fit_time = time.time() - start_time
    print(f"Fitting completed in {fit_time:.2f} seconds")
    
    # Get multioutput info
    info = model.get_multioutput_info()
    print(f"\nMultioutput info:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Make predictions
    print("\nMaking 30-day forecasts with 95% prediction intervals...")
    forecast = model.predict(h=30, level=95)
    
    print(f"\nPoint forecasts shape: {forecast.mean.shape}")
    print("\nFirst 5 forecast periods:")
    print(forecast.mean.head())
    
    print("\nPrediction intervals (first period):")
    for i, series in enumerate(df.columns):
        print(f"  {series}: {forecast.lower.iloc[0, i]:.2f} < {forecast.mean.iloc[0, i]:.2f} < {forecast.upper.iloc[0, i]:.2f}")
    
    return model, df, forecast


def example_2_wrapped_non_multioutput():
    """Example 2: Using RandomForest with MultiOutputRegressor wrapper."""
    print("\n" + "="*60)
    print("EXAMPLE 2: RandomForest with MultiOutputRegressor Wrapper")
    print("="*60)
    
    df = create_sample_data(n_samples=100, n_series=2)
    
    # Create RandomForest (not natively multioutput)
    rf = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
    
    # Wrap it for multioutput support
    multi_rf = MultiOutputRegressor(rf)
    
    # Create model
    model = ns.MultiOutputMTS(
        obj=multi_rf,
        lags=15,
        n_hidden_features=10,
        type_pi='gaussian',
        verbose=1
    )
    
    print("Fitting model...")
    model.fit(df)
    
    # Make predictions
    forecast = model.predict(h=20, level=90)
    
    print(f"\nForecast shape: {forecast.mean.shape}")
    print("\nModel info:")
    info = model.get_multioutput_info()
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    return model, df, forecast


def example_3_performance_comparison():
    """Example 3: Performance comparison between MTS and MultiOutputMTS."""
    print("\n" + "="*60)
    print("EXAMPLE 3: Performance Comparison")
    print("="*60)
    
    # Create larger dataset for performance test
    df = create_sample_data(n_samples=500, n_series=5)
    
    # Test with Ridge (native multioutput)
    ridge = Ridge(alpha=1.0)
    
    # Regular MTS
    print("\nTesting Regular MTS (loop-based)...")
    start = time.time()
    model_regular = ns.MTS(
        obj=ridge,
        lags=15,
        n_hidden_features=10,
        type_pi='gaussian',
        verbose=0
    )
    model_regular.fit(df)
    regular_time = time.time() - start
    print(f"Regular MTS fitting time: {regular_time:.4f} seconds")
    
    # MultiOutputMTS
    print("\nTesting MultiOutputMTS...")
    start = time.time()
    model_multi = ns.MultiOutputMTS(
        obj=ridge,
        lags=15,
        n_hidden_features=10,
        type_pi='gaussian',
        verbose=0
    )
    model_multi.fit(df)
    multi_time = time.time() - start
    print(f"MultiOutputMTS fitting time: {multi_time:.4f} seconds")
    
    print(f"\nSpeedup: {regular_time/multi_time:.2f}x faster")
    
    # Compare predictions
    forecast_regular = model_regular.predict(h=10)
    forecast_multi = model_multi.predict(h=10)
    
    print("\nPrediction comparison (RMSE between methods):")
    for i, series in enumerate(df.columns):
        diff = np.sqrt(np.mean((forecast_regular.mean.iloc[:, i] - forecast_multi.mean.iloc[:, i])**2))
        print(f"  {series}: {diff:.6f}")
    
    return regular_time, multi_time


def example_4_quantile_forecasts():
    """Example 4: Quantile forecasts."""
    print("\n" + "="*60)
    print("EXAMPLE 4: Quantile Forecasts")
    print("="*60)
    
    df = create_sample_data(n_samples=120, n_series=2)
    
    # For quantile regression, we need a quantile-compatible model
    from sklearn.linear_model import LinearRegression
    
    lr = LinearRegression()
    
    model = ns.MultiOutputMTS(
        obj=lr,
        lags=15,
        n_hidden_features=12,
        type_pi='kde',  # Using KDE for quantile simulation
        replications=500,
        verbose=1
    )
    
    model.fit(df)
    
    # Predict specific quantiles
    quantiles = [0.05, 0.25, 0.5, 0.75, 0.95]
    print(f"\nPredicting quantiles: {quantiles}")
    
    quantile_forecast = model.predict(h=20, quantiles=quantiles)
    
    print(f"\nQuantile forecast columns: {quantile_forecast.columns.tolist()}")
    print("\nFirst forecast period:")
    print(quantile_forecast.iloc[0].round(2))
    
    return model, df, quantile_forecast


def example_5_visualization(model, df, forecast):
    """Example 5: Visualization of results."""
    print("\n" + "="*60)
    print("EXAMPLE 5: Visualization")
    print("="*60)
    
    # Plot each series
    fig, axes = plt.subplots(len(df.columns), 1, figsize=(12, 4*len(df.columns)))
    if len(df.columns) == 1:
        axes = [axes]
    
    for idx, (ax, series_name) in enumerate(zip(axes, df.columns)):
        # Historical data
        ax.plot(df.index, df[series_name], 'b-', label='Historical', linewidth=2)
        
        # Forecast
        forecast_dates = forecast.mean.index
        ax.plot(forecast_dates, forecast.mean[series_name], 'r-', label='Forecast', linewidth=2)
        
        # Prediction intervals
        ax.fill_between(
            forecast_dates,
            forecast.lower[series_name],
            forecast.upper[series_name],
            alpha=0.2, color='red', label=f'{model.level_}% Prediction Interval'
        )
        
        # Add vertical line at forecast start
        ax.axvline(x=df.index[-1], color='gray', linestyle='--', alpha=0.5)
        
        ax.set_title(f'{series_name} - 30-Day Forecast')
        ax.set_xlabel('Date')
        ax.set_ylabel('Value')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('multioutput_mts_forecast.png', dpi=150, bbox_inches='tight')
    print("Plot saved as 'multioutput_mts_forecast.png'")
    plt.show()
    
    # Spaghetti plot for first series if simulations available
    if hasattr(forecast, 'sims') and forecast.sims is not None:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot historical
        ax.plot(df.index, df.iloc[:, 0], 'b-', label='Historical', linewidth=2)
        
        # Plot a sample of simulations
        n_sims_to_plot = min(20, len(forecast.sims))
        for i in range(n_sims_to_plot):
            ax.plot(forecast.mean.index, forecast.sims[i].iloc[:, 0], 
                   alpha=0.1, color='red')
        
        # Plot mean forecast
        ax.plot(forecast.mean.index, forecast.mean.iloc[:, 0], 
               'r-', label='Mean Forecast', linewidth=2)
        
        ax.axvline(x=df.index[-1], color='gray', linestyle='--', alpha=0.5)
        ax.set_title(f'{df.columns[0]} - Spaghetti Plot ({n_sims_to_plot} simulations)')
        ax.set_xlabel('Date')
        ax.set_ylabel('Value')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('multioutput_mts_spaghetti.png', dpi=150, bbox_inches='tight')
        print("Spaghetti plot saved as 'multioutput_mts_spaghetti.png'")
        plt.show()


# ============================================================
# MAIN EXECUTION
# ============================================================
if __name__ == "__main__":
    # Run examples and capture returns
    print("\nRunning MultiOutputMTS Examples...")
    print("="*60)
    
    # Example 1: Ridge with KDE intervals
    model1, df1, forecast1 = example_1_native_multioutput()
    
    # Example 2: RandomForest with Gaussian intervals
    model2, df2, forecast2 = example_2_wrapped_non_multioutput()
    
    # Example 3: Performance comparison
    regular_time, multi_time = example_3_performance_comparison()
    
    # Example 4: Quantile forecasts
    model4, df4, quantile_forecast = example_4_quantile_forecasts()
    
    # Example 5: Visualization (using results from Example 1)
    example_5_visualization(model1, df1, forecast1)
    
    print("\n" + "="*60)
    print("All examples completed successfully!")
    print("="*60)
    print(f"\nSummary:")
    print(f"  - Example 1: Ridge regression with KDE intervals")
    print(f"  - Example 2: RandomForest with Gaussian intervals")
    print(f"  - Example 3: Speedup = {regular_time/multi_time:.2f}x")
    print(f"  - Example 4: Quantile forecasts generated")
    print(f"  - Example 5: Visualizations saved")