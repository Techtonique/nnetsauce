import os 
import numpy as np 
import nnetsauce as ns
from nnetsauce.rff import RandomFourierFeaturesRidge
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from time import time
import matplotlib.pyplot as plt

print(f"\n ----- Running: {os.path.basename(__file__)}... ----- \n")


def bayesian_random_fourier_features_ridge():
    """Demonstrate the RFF class with both standard and Bayesian versions"""
    
    # Generate synthetic data
    np.random.seed(42)
    n_train = 100
    n_test = 200
    
    def true_function(x):
        return np.sin(2 * np.pi * x) + 0.3 * np.cos(4 * np.pi * x)
    
    X_train = np.random.uniform(0, 1, n_train).reshape(-1, 1)
    y_train = true_function(X_train).ravel() + np.random.normal(0, 0.1, n_train)
    
    X_test = np.linspace(0, 1, n_test).reshape(-1, 1)
    y_test = true_function(X_test).ravel()
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Test different configurations
    configs = [
        {'n_features': 50, 'method': 'standard', 'title': 'Standard RFF (50 features)'},
        {'n_features': 200, 'method': 'standard', 'title': 'Standard RFF (200 features)'},
        {'n_features': 50, 'method': 'bayesian', 'title': 'Bayesian RFF (50 features)'},
        {'n_features': 200, 'method': 'bayesian', 'title': 'Bayesian RFF (200 features)'},
    ]
    
    for idx, config in enumerate(configs):
        ax = axes[idx // 2, idx % 2]
    
        # Create and fit model
        model = RandomFourierFeaturesRidge(
            n_features=config['n_features'],
            gamma=1.0,
            alpha=1e-4,
            include_bias=True
        )
    
        model.fit(X_train, y_train, method=config['method'])
    
        # Make predictions with uncertainty for Bayesian models
        if config['method'] == 'bayesian':
            y_pred, y_std = model.predict(X_test, return_std=True)
    
            # Plot with uncertainty bands
            ax.fill_between(
                X_test.ravel(),
                (y_pred - 2 * y_std).ravel(),
                (y_pred + 2 * y_std).ravel(),
                alpha=0.3,
                color='orange',
                label='±2σ'
            )
        else:
            y_pred = model.predict(X_test)
    
        # Plot results
        ax.scatter(X_train, y_train, alpha=0.5, label='Training data', s=20)
        ax.plot(X_test, y_test, 'k-', linewidth=2, label='True function')
        ax.plot(X_test, y_pred, 'r--', linewidth=2, label='Prediction')
    
        ax.set_title(config['title'])
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
    
        # Calculate and display metrics
        train_pred = model.predict(X_train)
        train_mse = np.mean((y_train - train_pred.ravel()) ** 2)
        test_mse = np.mean((y_test - y_pred.ravel()) ** 2)
    
        ax.text(0.05, 0.95, f'Train MSE: {train_mse:.4f}\nTest MSE: {test_mse:.4f}',
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('rff_bayesian_demo.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Demonstrate posterior sampling
    print("\n" + "="*60)
    print("Demonstrating Posterior Sampling from Bayesian RFF")
    print("="*60)
    
    model_bayes = RandomFourierFeaturesRidge(n_features=100, gamma=1.0, alpha=1e-4)
    model_bayes.fit(X_train, y_train, method='bayesian')
    
    # Sample from posterior
    samples = model_bayes.sample_posterior(X_test, n_samples=5)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    for i in range(5):
        ax.plot(X_test, samples[i], '--', alpha=0.6, label=f'Sample {i+1}')
    
    ax.plot(X_test, y_test, 'k-', linewidth=2, label='True function')
    ax.scatter(X_train, y_train, alpha=0.5, label='Training data', s=20)
    
    y_pred, y_std = model_bayes.predict(X_test, return_std=True)
    ax.plot(X_test, y_pred, 'r-', linewidth=2, label='Predictive mean')
    ax.fill_between(
        X_test.ravel(),
        (y_pred - 2 * y_std).ravel(),
        (y_pred + 2 * y_std).ravel(),
        alpha=0.2, color='red', label='±2σ'
    )
    
    ax.set_title('Posterior Samples from Bayesian RFF')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('rff_posterior_samples.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Print model evidence
    print(f"\nLog Marginal Likelihood: {model_bayes.log_marginal_likelihood():.2f}")
    
    return model_bayes


if __name__ == "__main__":
    model_bayes = bayesian_random_fourier_features_ridge()
    

