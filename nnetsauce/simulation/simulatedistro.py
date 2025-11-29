import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, norm
from scipy.interpolate import interp1d


def simulate_distribution(
    data, method="bootstrap", num_samples=1000, seed=123, **kwargs
):
    """
    Simulate the distribution of an input vector using various methods.

    Parameters:
        data (array-like): Input vector of data.
        method (str): Method for simulation:
                      - 'bootstrap': Bootstrap resampling.
                      - 'kde': Kernel Density Estimation.
                      - 'normal': Normal distribution.
                      - 'ecdf': Empirical CDF-based sampling.
                      - 'permutation': Permutation resampling.
                      - 'smooth-bootstrap': Smoothed bootstrap with added noise.
        num_samples (int): Number of samples to generate.
        seed (int): Random seed for reproducibility.
        kwargs: Additional parameters for specific methods:
                - kde_bandwidth (str or float): Bandwidth for KDE ('scott', 'silverman', or float).
                - dist (str): Parametric distribution type ('normal').
                - noise_std (float): Noise standard deviation for smoothed bootstrap.

    Returns:
        np.ndarray: Simulated distribution samples.
    """
    assert method in [
        "bootstrap",
        "kde",
        "parametric",
        "ecdf",
        "permutation",
        "smooth-bootstrap",
    ], f"Unknown method '{method}'. Choose from 'bootstrap', 'kde', 'parametric', 'ecdf', 'permutation', or 'smooth_bootstrap'."

    data = np.array(data)

    np.random.seed(seed)

    if method == "bootstrap":
        simulated_data = np.random.choice(data, size=num_samples, replace=True)

    elif method == "kde":
        kde_bandwidth = kwargs.get("kde_bandwidth", "scott")
        kde = gaussian_kde(data, bw_method=kde_bandwidth)
        simulated_data = kde.resample(num_samples).flatten()

    elif method == "normal":
        mean, std = np.mean(data), np.std(data)
        simulated_data = np.random.normal(mean, std, size=num_samples)

    elif method == "ecdf":
        data = np.sort(data)
        ecdf_y = np.arange(1, len(data) + 1) / len(data)
        inverse_cdf = interp1d(
            ecdf_y, data, bounds_error=False, fill_value=(data[0], data[-1])
        )
        random_uniform = np.random.uniform(0, 1, size=num_samples)
        simulated_data = inverse_cdf(random_uniform)

    elif method == "permutation":
        simulated_data = np.random.permutation(data)
        while len(simulated_data) < num_samples:
            simulated_data = np.concatenate(
                [simulated_data, np.random.permutation(data)]
            )
        simulated_data = simulated_data[:num_samples]

    elif method == "smooth_bootstrap":
        noise_std = kwargs.get("noise_std", 0.1)
        bootstrap_samples = np.random.choice(
            data, size=num_samples, replace=True
        )
        noise = np.random.normal(0, noise_std, size=num_samples)
        simulated_data = bootstrap_samples + noise

    else:
        raise ValueError(
            f"Unknown method '{method}'. Choose from 'bootstrap', 'kde', 'parametric', 'ecdf', 'permutation', or 'smooth_bootstrap'."
        )

    return simulated_data


def simulate_replications(
    data, method="kde", num_replications=10, n_obs=None, seed=123, **kwargs
):
    """
    Create multiple replications of the input's distribution using a specified simulation method.

    Parameters:
        data (array-like): Input vector of data.
        method (str): Method for simulation:
                      - 'bootstrap': Bootstrap resampling.
                      - 'kde': Kernel Density Estimation.
                      - 'normal': Parametric distribution fitting.
                      - 'ecdf': Empirical CDF-based sampling.
                      - 'permutation': Permutation resampling.
                      - 'smooth_bootstrap': Smoothed bootstrap with added noise.
        num_samples (int): Number of samples in each replication.
        num_replications (int): Number of replications to generate.
        n_obs (int): Number of observations to generate for each replication.
        seed (int): Random seed for reproducibility.
        kwargs: Additional parameters for specific methods.

    Returns:
        pd.DataFrame: A DataFrame where each column represents a replication.
    """

    num_samples = len(data)

    replications = []

    for _ in range(num_replications):
        simulated_data = simulate_distribution(
            data, method=method, num_samples=num_samples, seed=seed, **kwargs
        )
        replications.append(simulated_data)

    # Combine replications into a DataFrame
    replications_df = pd.DataFrame(replications).transpose()
    replications_df.columns = [
        f"Replication_{i+1}" for i in range(num_replications)
    ]

    # If n_obs is specified, sample n_obs from each replication
    if n_obs is not None:
        replications_df = replications_df.sample(
            n=n_obs, replace=True, random_state=42
        ).reset_index(drop=True)
        return replications_df.values

    return replications_df.values
