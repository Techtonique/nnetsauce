import numpy as np

try:
    import jax.numpy as jnp
    from jax import device_put
    from jax.numpy.linalg import inv as jinv
except ImportError:
    pass


def get_beta(X, y, backend="cpu"):
    if backend == "cpu":
        main_component = np.linalg.solve(X.T @ X, X.T)
    else:
        device_put(X)
        device_put(y)
        main_component = jnp.linalg.solve(X.T @ X, X.T)
    H = X @ main_component
    return main_component @ y, H


def get_beta_ridge(X, y, lam, backend="cpu"):
    if backend == "cpu":
        return np.linalg.inv(X.T @ X + lam * np.eye(X.shape[1])) @ X.T @ y
    else:
        device_put(X)
        device_put(y)
        XTX = X.T @ X
        Xty = X.T @ y
        I = np.eye(X.shape[1])
        device_put(XTX)
        device_put(Xty)
        device_put(I)
        return jinv(XTX + lam * I) @ Xty


def get_best_beta(X, y, lambdas, backend="cpu"):
    # Number of features
    n_features = X.shape[1]
    # Compute X^T X and X^T y
    if backend == "gpu":
        device_put(X)
        device_put(y)

    Xt = X.T
    XtX = Xt @ X
    Xty = Xt @ y
    I = np.eye(n_features)

    if backend == "gpu":
        device_put(Xt)
        device_put(XtX)
        device_put(Xty)
        device_put(I)
        hat_matrices_diags = [
            (X @ jinv(XtX + lam * I) @ Xt).diagonal() for lam in lambdas
        ]
    else:
        hat_matrices_diags = [
            (X @ np.linalg.inv(XtX + lam * I) @ Xt).diagonal()
            for lam in lambdas
        ]
    train_errors = [
        np.linalg.norm(
            (y - X @ get_beta_ridge(X, y, lam=lam, backend=backend))
            / (1 - hat_matrices_diags[i])
        )
        for i, lam in enumerate(lambdas)
    ]
    train_errors = np.asarray(train_errors)
    best_lam = lambdas[np.argmin(train_errors)]
    # best coefficients
    return (
        get_beta_ridge(X, y, lam=best_lam, backend=backend),
        train_errors,
        best_lam,
    )
