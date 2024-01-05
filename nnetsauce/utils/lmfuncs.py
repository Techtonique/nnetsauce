import numpy as np
import platform
from numpy import linalg as la

if platform.system() in ("Linux", "Darwin"):
    import jax.numpy as jnp
    from jax.numpy import linalg as jla

# import .matrixops as mo
from . import matrixops as mo


# computes beta_hat = (t(x)%*%x + lam*I)^{-1}%*%t(x)%*%y
def beta_hat(x, y, lam=None, backend="cpu"):
    # assert on dimensions
    if lam is None:
        return mo.safe_sparse_dot(
            a=inv_penalized_cov(x=x, backend=backend),
            b=mo.crossprod(x=x, y=y, backend=backend),
            backend=backend,
        )

    return mo.safe_sparse_dot(
        a=inv_penalized_cov(x=x, lam=lam, backend=backend),
        b=mo.crossprod(x=x, y=y, backend=backend),
        backend=backend,
    )


# computes (t(x)%*%x + lam*I)^{-1}
def inv_penalized_cov(x, lam=None, backend="cpu"):
    # assert on dimensions
    sys_platform = platform.system()
    if backend in ("gpu", "tpu") and (sys_platform in ("Linux", "Darwin")):
        if lam is None:
            return jla.inv(mo.crossprod(x=x, backend=backend))
        return jla.inv(
            mo.crossprod(x=x, backend=backend) + lam * jnp.eye(x.shape[1])
        )

    if lam is None:
        return la.inv(mo.crossprod(x))
    return la.inv(mo.crossprod(x) + lam * np.eye(x.shape[1]))


# linear regression with no regularization
def beta_Sigma_hat(
    X=None,
    y=None,
    fit_intercept=False,
    X_star=None,  # for preds
    return_cov=True,  # confidence interval required for preds?
    beta_hat_=None,  # for prediction only (X_star is not None)
    Sigma_hat_=None,
    backend="cpu",
):  # for prediction only (X_star is not None)
    if (X is not None) & (y is not None):  # fit
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)

        if X_star is not None:
            if len(X_star.shape) == 1:
                X_star = X_star.reshape(-1, 1)

        n, p = X.shape

        if fit_intercept == True:
            X = mo.cbind(x=np.ones(n), y=X, backend=backend)
            if X_star is not None:
                X_star = mo.cbind(
                    x=np.ones(X_star.shape[0]), y=X_star, backend=backend
                )

        Cn = inv_penalized_cov(x=X, backend=backend)

        if return_cov == True:
            if X_star is None:
                beta_hat_ = mo.safe_sparse_dot(
                    a=Cn,
                    b=mo.crossprod(x=X, y=y, backend=backend),
                    backend=backend,
                )

                Sigma_hat_ = np.eye(X.shape[1]) - mo.safe_sparse_dot(
                    a=Cn, b=mo.crossprod(x=X, backend=backend), backend=backend
                )
                temp = mo.safe_sparse_dot(
                    a=Cn,
                    b=mo.tcrossprod(x=Sigma_hat_, y=X, backend=backend),
                    backend=backend,
                )
                smoothing_matrix = mo.safe_sparse_dot(
                    a=X, b=temp, backend=backend
                )
                y_hat = mo.safe_sparse_dot(
                    a=smoothing_matrix, b=y, backend=backend
                )

                return {
                    "beta_hat": beta_hat_,
                    "Sigma_hat": Sigma_hat_,
                    "GCV": np.mean(
                        ((y - y_hat) / (1 - np.trace(smoothing_matrix) / n))
                        ** 2
                    ),
                }

            else:
                if beta_hat_ is None:
                    beta_hat_ = mo.safe_sparse_dot(
                        a=Cn,
                        b=mo.crossprod(X, y, backend=backend),
                        backend=backend,
                    )

                if Sigma_hat_ is None:
                    Sigma_hat_ = np.eye(X.shape[1]) - mo.safe_sparse_dot(
                        a=Cn,
                        b=mo.crossprod(x=X, backend=backend),
                        backend=backend,
                    )

                temp = mo.safe_sparse_dot(
                    a=Cn,
                    b=mo.tcrossprod(Sigma_hat_, X, backend=backend),
                    backend=backend,
                )
                smoothing_matrix = mo.safe_sparse_dot(
                    a=X, b=temp, backend=backend
                )
                y_hat = mo.safe_sparse_dot(
                    a=smoothing_matrix, b=y, backend=backend
                )

                return {
                    "beta_hat": beta_hat_,
                    "Sigma_hat": Sigma_hat_,
                    "GCV": np.mean(
                        ((y - y_hat) / (1 - np.trace(smoothing_matrix) / n))
                        ** 2
                    ),
                    "preds": mo.safe_sparse_dot(
                        a=X_star, b=beta_hat_, backend=backend
                    ),
                    "preds_std": np.sqrt(
                        np.diag(
                            mo.safe_sparse_dot(
                                a=X_star,
                                b=mo.tcrossprod(
                                    x=Sigma_hat_, y=X_star, backend=backend
                                ),
                                backend=backend,
                            )
                        )
                    ),
                }

        else:  # return_cov == False
            if X_star is None:
                beta_hat_ = mo.safe_sparse_dot(
                    a=Cn,
                    b=mo.crossprod(x=X, y=y, backend=backend),
                    backend=backend,
                )
                Sigma_hat_ = np.eye(X.shape[1]) - mo.safe_sparse_dot(
                    a=Cn, b=mo.crossprod(x=X, backend=backend), backend=backend
                )
                temp = mo.safe_sparse_dot(
                    a=Cn,
                    b=mo.tcrossprod(x=Sigma_hat_, y=X, backend=backend),
                    backend=backend,
                )
                smoothing_matrix = mo.safe_sparse_dot(
                    a=X, b=temp, backend=backend
                )
                y_hat = mo.safe_sparse_dot(
                    a=smoothing_matrix, b=y, backend=backend
                )

                return {
                    "beta_hat": beta_hat_,
                    "GCV": np.mean(
                        ((y - y_hat) / (1 - np.trace(smoothing_matrix) / n))
                        ** 2
                    ),
                }

            else:
                if beta_hat_ is None:
                    beta_hat_ = beta_hat(X, y)
                    Sigma_hat_ = np.eye(X.shape[1]) - mo.safe_sparse_dot(
                        a=Cn,
                        b=mo.crossprod(x=X, backend=backend),
                        backend=backend,
                    )
                    temp = mo.safe_sparse_dot(
                        a=Cn,
                        b=mo.tcrossprod(x=Sigma_hat_, y=X, backend=backend),
                        backend=backend,
                    )
                    smoothing_matrix = mo.safe_sparse_dot(
                        a=X, b=temp, backend=backend
                    )
                    y_hat = mo.safe_sparse_dot(
                        a=smoothing_matrix, b=y, backend=backend
                    )

                return {
                    "beta_hat": beta_hat_,
                    "GCV": np.mean(
                        ((y - y_hat) / (1 - np.trace(smoothing_matrix) / n))
                        ** 2
                    ),
                    "preds": mo.safe_sparse_dot(
                        a=X_star, b=beta_hat_, backend=backend
                    ),
                }

    else:  # X is None | y is None  # predict
        assert (beta_hat_ is not None) & (X_star is not None)

        if return_cov == True:
            assert Sigma_hat_ is not None

            return {
                "preds": mo.safe_sparse_dot(
                    a=X_star, b=beta_hat_, backend=backend
                ),
                "preds_std": np.sqrt(
                    np.diag(
                        mo.safe_sparse_dot(
                            a=X_star,
                            b=mo.tcrossprod(
                                x=Sigma_hat_, y=X_star, backend=backend
                            ),
                            backend=backend,
                        )
                    )
                ),
            }

        else:
            return {
                "preds": mo.safe_sparse_dot(
                    a=X_star, b=beta_hat_, backend=backend
                )
            }


# beta and Sigma in Bayesian Ridge Regression 1
# without intercept! without intercept! without intercept!
def beta_Sigma_hat_rvfl(
    X=None,
    y=None,
    s=0.1,
    sigma=0.05,
    fit_intercept=False,
    X_star=None,  # for preds
    return_cov=True,  # confidence interval required for preds?
    beta_hat_=None,  # for prediction only (X_star is not None)
    Sigma_hat_=None,
    backend="cpu",
):  # for prediction only (X_star is not None)
    if (X is not None) & (y is not None):  # fit
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)

        if X_star is not None:
            if len(X_star.shape) == 1:
                X_star = X_star.reshape(-1, 1)

        n, p = X.shape

        if fit_intercept == True:
            X = mo.cbind(np.ones(n), X, backend=backend)
            if X_star is not None:
                X_star = mo.cbind(
                    x=np.ones(X_star.shape[0]), y=X_star, backend=backend
                )

        s2 = s**2
        lambda_ = (sigma**2) / s2
        Cn = inv_penalized_cov(X, lam=lambda_)

        if return_cov == True:
            if X_star is None:
                beta_hat_ = mo.safe_sparse_dot(
                    a=Cn,
                    b=mo.crossprod(x=X, y=y, backend=backend),
                    backend=backend,
                )
                Sigma_hat_ = s2 * (
                    np.eye(X.shape[1])
                    - mo.safe_sparse_dot(
                        a=Cn,
                        b=mo.crossprod(x=X, backend=backend),
                        backend=backend,
                    )
                )
                temp = mo.safe_sparse_dot(
                    a=Cn,
                    b=mo.tcrossprod(x=Sigma_hat_, y=X, backend=backend),
                    backend=backend,
                )
                smoothing_matrix = mo.safe_sparse_dot(
                    a=X, b=temp, backend=backend
                )
                y_hat = mo.safe_sparse_dot(
                    a=smoothing_matrix, b=y, backend=backend
                )

                return {
                    "beta_hat": beta_hat_,
                    "Sigma_hat": Sigma_hat_,
                    "GCV": np.mean(
                        ((y - y_hat) / (1 - np.trace(smoothing_matrix) / n))
                        ** 2
                    ),
                }

            else:
                if beta_hat_ is None:
                    beta_hat_ = mo.safe_sparse_dot(
                        a=Cn,
                        b=mo.crossprod(x=X, y=y, backend=backend),
                        backend=backend,
                    )

                if Sigma_hat_ is None:
                    Sigma_hat_ = s2 * (
                        np.eye(X.shape[1])
                        - mo.safe_sparse_dot(
                            a=Cn,
                            b=mo.crossprod(x=X, backend=backend),
                            backend=backend,
                        )
                    )

                temp = mo.safe_sparse_dot(
                    a=Cn,
                    b=mo.tcrossprod(x=Sigma_hat_, y=X, backend=backend),
                    backend=backend,
                )
                smoothing_matrix = mo.safe_sparse_dot(
                    a=X, b=temp, backend=backend
                )
                y_hat = mo.safe_sparse_dot(
                    a=smoothing_matrix, b=y, backend=backend
                )

                return {
                    "beta_hat": beta_hat_,
                    "Sigma_hat": Sigma_hat_,
                    "GCV": np.mean(
                        ((y - y_hat) / (1 - np.trace(smoothing_matrix) / n))
                        ** 2
                    ),
                    "preds": mo.safe_sparse_dot(
                        a=X_star, b=beta_hat_, backend=backend
                    ),
                    "preds_std": np.sqrt(
                        np.diag(
                            mo.safe_sparse_dot(
                                a=X_star,
                                b=mo.tcrossprod(
                                    x=Sigma_hat_, y=X_star, backend=backend
                                ),
                                backend=backend,
                            )
                            + (sigma**2) * np.eye(X_star.shape[0])
                        )
                    ),
                }

        else:  # return_cov == False
            if X_star is None:
                beta_hat_ = mo.safe_sparse_dot(
                    a=Cn,
                    b=mo.crossprod(x=X, y=y, backend=backend),
                    backend=backend,
                )
                Sigma_hat_ = s2 * (
                    np.eye(X.shape[1])
                    - mo.safe_sparse_dot(
                        a=Cn, b=mo.crossprod(X), backend=backend
                    )
                )
                temp = mo.safe_sparse_dot(
                    a=Cn,
                    b=mo.tcrossprod(x=Sigma_hat_, y=X, backend=backend),
                    backend=backend,
                )
                smoothing_matrix = mo.safe_sparse_dot(
                    a=X, b=temp, backend=backend
                )
                y_hat = mo.safe_sparse_dot(
                    a=smoothing_matrix, b=y, backend=backend
                )

                return {
                    "beta_hat": beta_hat_,
                    "GCV": np.mean(
                        ((y - y_hat) / (1 - np.trace(smoothing_matrix) / n))
                        ** 2
                    ),
                }

            else:
                if beta_hat_ is None:
                    beta_hat_ = beta_hat(X, y, lam=lambda_, backend=backend)
                    Sigma_hat_ = s2 * (
                        np.eye(X.shape[1])
                        - mo.safe_sparse_dot(
                            a=Cn,
                            b=mo.crossprod(x=X, backend=backend),
                            backend=backend,
                        )
                    )
                    temp = mo.safe_sparse_dot(
                        a=Cn,
                        b=mo.tcrossprod(x=Sigma_hat_, y=X, backend=backend),
                        backend=backend,
                    )
                    smoothing_matrix = mo.safe_sparse_dot(
                        a=X, b=temp, backend=backend
                    )
                    y_hat = mo.safe_sparse_dot(
                        a=smoothing_matrix, b=y, backend=backend
                    )

                return {
                    "beta_hat": beta_hat_,
                    "GCV": np.mean(
                        ((y - y_hat) / (1 - np.trace(smoothing_matrix) / n))
                        ** 2
                    ),
                    "preds": mo.safe_sparse_dot(
                        a=X_star, b=beta_hat_, backend=backend
                    ),
                }

    else:  # X is None | y is None  # predict
        assert (beta_hat_ is not None) & (X_star is not None)

        if return_cov == True:
            assert Sigma_hat_ is not None

            return {
                "preds": mo.safe_sparse_dot(X_star, beta_hat_, backend=backend),
                "preds_std": np.sqrt(
                    np.diag(
                        mo.safe_sparse_dot(
                            a=X_star,
                            b=mo.tcrossprod(
                                x=Sigma_hat_, y=X_star, backend=backend
                            ),
                            backend=backend,
                        )
                        + (sigma**2) * np.eye(X_star.shape[0])
                    )
                ),
            }

        else:
            return {
                "preds": mo.safe_sparse_dot(
                    a=X_star, b=beta_hat_, backend=backend
                )
            }


# beta and Sigma in Bayesian Ridge Regression 2
# without intercept! without intercept! without intercept!
def beta_Sigma_hat_rvfl2(
    X=None,
    y=None,
    Sigma=None,
    sigma=0.05,
    fit_intercept=False,
    X_star=None,  # check when dim = 1 # check when dim = 1
    return_cov=True,
    beta_hat_=None,  # for prediction only (X_star is not None)
    Sigma_hat_=None,
    backend="cpu",
):  # for prediction only (X_star is not None)
    if (X is not None) & (y is not None):
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)

        n, p = X.shape

        if Sigma is None:
            if fit_intercept == True:
                Sigma = np.eye(p + 1)
            else:
                Sigma = np.eye(p)

        if X_star is not None:
            if len(X_star.shape) == 1:
                X_star = X_star.reshape(-1, 1)

        if fit_intercept == True:
            X = mo.cbind(np.ones(n), X)
            Cn = (
                la.inv(
                    mo.safe_sparse_dot(
                        a=Sigma,
                        b=mo.crossprod(x=X, backend=backend),
                        backend="cpu",
                    )
                    + (sigma**2) * np.eye(p + 1)
                )
                if backend == "cpu"
                else jla.inv(
                    mo.safe_sparse_dot(
                        a=Sigma,
                        b=mo.crossprod(x=X, backend=backend),
                        backend=backend,
                    )
                    + (sigma**2) * np.eye(p + 1)
                )
            )

            if X_star is not None:
                X_star = mo.cbind(
                    x=np.ones(X_star.shape[0]), y=X_star, backend=backend
                )
        else:
            # rename to invCn
            Cn = (
                la.inv(
                    mo.safe_sparse_dot(
                        a=Sigma,
                        b=mo.crossprod(x=X, backend=backend),
                        backend="cpu",
                    )
                    + (sigma**2) * np.eye(p)
                )
                if backend == "cpu"
                else jla.inv(
                    mo.safe_sparse_dot(
                        a=Sigma,
                        b=mo.crossprod(x=X, backend=backend),
                        backend=backend,
                    )
                    + (sigma**2) * np.eye(p)
                )
            )

        temp = mo.safe_sparse_dot(
            a=Cn,
            b=mo.tcrossprod(x=Sigma, y=X, backend=backend),
            backend=backend,
        )
        smoothing_matrix = mo.safe_sparse_dot(a=X, b=temp, backend=backend)
        y_hat = mo.safe_sparse_dot(a=smoothing_matrix, b=y, backend=backend)

        if return_cov == True:
            if X_star is None:
                return {
                    "beta_hat": mo.safe_sparse_dot(
                        a=temp, b=y, backend=backend
                    ),
                    "Sigma_hat": Sigma
                    - mo.safe_sparse_dot(
                        a=temp,
                        b=mo.safe_sparse_dot(a=X, b=Sigma, backend=backend),
                        backend=backend,
                    ),
                    "GCV": np.mean(
                        ((y - y_hat) / (1 - np.trace(smoothing_matrix) / n))
                        ** 2
                    ),
                }

            else:
                if beta_hat_ is None:
                    beta_hat_ = mo.safe_sparse_dot(a=temp, b=y, backend=backend)

                if Sigma_hat_ is None:
                    Sigma_hat_ = Sigma - mo.safe_sparse_dot(
                        a=temp,
                        b=mo.safe_sparse_dot(a=X, b=Sigma, backend=backend),
                        backend=backend,
                    )

                return {
                    "beta_hat": beta_hat_,
                    "Sigma_hat": Sigma_hat_,
                    "GCV": np.mean(
                        ((y - y_hat) / (1 - np.trace(smoothing_matrix) / n))
                        ** 2
                    ),
                    "preds": mo.safe_sparse_dot(
                        a=X_star, b=beta_hat_, backend=backend
                    ),
                    "preds_std": np.sqrt(
                        np.diag(
                            mo.safe_sparse_dot(
                                a=X_star,
                                b=mo.tcrossprod(
                                    x=Sigma_hat_, y=X_star, backend=backend
                                ),
                                backend=backend,
                            )
                            + (sigma**2) * np.eye(X_star.shape[0])
                        )
                    ),
                }

        else:  # return_cov == False
            if X_star is None:
                return {
                    "beta_hat": mo.safe_sparse_dot(
                        a=temp, b=y, backend=backend
                    ),
                    "GCV": np.mean(
                        ((y - y_hat) / (1 - np.trace(smoothing_matrix) / n))
                        ** 2
                    ),
                }

            else:
                if beta_hat_ is None:
                    beta_hat_ = mo.safe_sparse_dot(a=temp, b=y, backend=backend)

                return {
                    "beta_hat": beta_hat_,
                    "GCV": np.mean(
                        ((y - y_hat) / (1 - np.trace(smoothing_matrix) / n))
                        ** 2
                    ),
                    "preds": mo.safe_sparse_dot(
                        a=X_star, b=beta_hat_, backend=backend
                    ),
                }

    else:  # (X is None) | (y is None) # predict
        assert (beta_hat_ is not None) & (X_star is not None)

        if return_cov == True:
            assert Sigma_hat_ is not None

            return {
                "preds": mo.safe_sparse_dot(
                    a=X_star, b=beta_hat_, backend=backend
                ),
                "preds_std": np.sqrt(
                    np.diag(
                        mo.safe_sparse_dot(
                            a=X_star,
                            b=mo.tcrossprod(
                                Sigma_hat_, X_star, backend=backend
                            ),
                            backend=backend,
                        )
                        + (sigma**2) * np.eye(X_star.shape[0])
                    )
                ),
            }

        else:
            return {
                "preds": mo.safe_sparse_dot(
                    a=X_star, b=beta_hat_, backend=backend
                )
            }
