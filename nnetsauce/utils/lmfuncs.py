import numpy as np
from numpy import linalg as la

# import .matrixops as mo
from . import matrixops as mo


# computes beta_hat = (t(x)%*%x + lam*I)^{-1}%*%t(x)%*%y
def beta_hat(x, y, lam=None):
    # assert on dimensions
    if lam is None:
        return np.dot(
            inv_penalized_cov(x), mo.crossprod(x, y)
        )
    else:
        return np.dot(
            inv_penalized_cov(x, lam), mo.crossprod(x, y)
        )


# computes (t(x)%*%x + lam*I)^{-1}
def inv_penalized_cov(x, lam=None):
    # assert on dimensions
    if lam is None:
        return la.inv(mo.crossprod(x))
    else:
        return la.inv(
            mo.crossprod(x) + lam * np.eye(x.shape[1])
        )


# linear regression with no regularization
def beta_Sigma_hat(
    X=None,
    y=None,
    fit_intercept=False,
    X_star=None,  # for preds
    return_cov=True,  # confidence interval required for preds?
    beta_hat_=None,  # for prediction only (X_star is not None)
    Sigma_hat_=None,
):  # for prediction only (X_star is not None)

    if (X is not None) & (y is not None):  # fit

        if len(X.shape) == 1:
            X = X.reshape(-1, 1)

        if X_star is not None:
            if len(X_star.shape) == 1:
                X_star = X_star.reshape(-1, 1)

        n, p = X.shape

        if fit_intercept == True:
            X = mo.cbind(np.ones(n), X)
            if X_star is not None:
                X_star = mo.cbind(
                    np.ones(X_star.shape[0]), X_star
                )

        Cn = inv_penalized_cov(X)

        if return_cov == True:

            if X_star is None:

                beta_hat_ = np.dot(Cn, mo.crossprod(X, y))
                Sigma_hat_ = np.eye(X.shape[1]) - np.dot(
                    Cn, mo.crossprod(X)
                )
                temp = np.dot(
                    Cn, mo.tcrossprod(Sigma_hat_, X)
                )
                smoothing_matrix = np.dot(X, temp)
                y_hat = np.dot(smoothing_matrix, y)

                return {
                    "beta_hat": beta_hat_,
                    "Sigma_hat": Sigma_hat_,
                    "GCV": np.mean(
                        (
                            (y - y_hat)
                            / (
                                1
                                - np.trace(smoothing_matrix)
                                / n
                            )
                        )
                        ** 2
                    ),
                }

            else:

                if beta_hat_ is None:
                    beta_hat_ = np.dot(
                        Cn, mo.crossprod(X, y)
                    )

                if Sigma_hat_ is None:
                    Sigma_hat_ = np.eye(
                        X.shape[1]
                    ) - np.dot(Cn, mo.crossprod(X))

                temp = np.dot(
                    Cn, mo.tcrossprod(Sigma_hat_, X)
                )
                smoothing_matrix = np.dot(X, temp)
                y_hat = np.dot(smoothing_matrix, y)

                return {
                    "beta_hat": beta_hat_,
                    "Sigma_hat": Sigma_hat_,
                    "GCV": np.mean(
                        (
                            (y - y_hat)
                            / (
                                1
                                - np.trace(smoothing_matrix)
                                / n
                            )
                        )
                        ** 2
                    ),
                    "preds": np.dot(X_star, beta_hat_),
                    "preds_std": np.sqrt(
                        np.diag(
                            np.dot(
                                X_star,
                                mo.tcrossprod(
                                    Sigma_hat_, X_star
                                ),
                            )
                        )
                    ),
                }

        else:  # return_cov == False

            if X_star is None:

                beta_hat_ = np.dot(Cn, mo.crossprod(X, y))
                Sigma_hat_ = np.eye(X.shape[1]) - np.dot(
                    Cn, mo.crossprod(X)
                )
                temp = np.dot(
                    Cn, mo.tcrossprod(Sigma_hat_, X)
                )
                smoothing_matrix = np.dot(X, temp)
                y_hat = np.dot(smoothing_matrix, y)

                return {
                    "beta_hat": beta_hat_,
                    "GCV": np.mean(
                        (
                            (y - y_hat)
                            / (
                                1
                                - np.trace(smoothing_matrix)
                                / n
                            )
                        )
                        ** 2
                    ),
                }

            else:

                if beta_hat_ is None:
                    beta_hat_ = beta_hat(X, y)
                    Sigma_hat_ = np.eye(
                        X.shape[1]
                    ) - np.dot(Cn, mo.crossprod(X))
                    temp = np.dot(
                        Cn, mo.tcrossprod(Sigma_hat_, X)
                    )
                    smoothing_matrix = np.dot(X, temp)
                    y_hat = np.dot(smoothing_matrix, y)

                return {
                    "beta_hat": beta_hat_,
                    "GCV": np.mean(
                        (
                            (y - y_hat)
                            / (
                                1
                                - np.trace(smoothing_matrix)
                                / n
                            )
                        )
                        ** 2
                    ),
                    "preds": np.dot(X_star, beta_hat_),
                }

    else:  # X is None | y is None  # predict

        assert (beta_hat_ is not None) & (
            X_star is not None
        )

        if return_cov == True:

            assert Sigma_hat_ is not None

            return {
                "preds": np.dot(X_star, beta_hat_),
                "preds_std": np.sqrt(
                    np.diag(
                        np.dot(
                            X_star,
                            mo.tcrossprod(
                                Sigma_hat_, X_star
                            ),
                        )
                    )
                ),
            }

        else:

            return {"preds": np.dot(X_star, beta_hat_)}


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
):  # for prediction only (X_star is not None)

    if (X is not None) & (y is not None):  # fit

        if len(X.shape) == 1:
            X = X.reshape(-1, 1)

        if X_star is not None:
            if len(X_star.shape) == 1:
                X_star = X_star.reshape(-1, 1)

        n, p = X.shape

        if fit_intercept == True:
            X = mo.cbind(np.ones(n), X)
            if X_star is not None:
                X_star = mo.cbind(
                    np.ones(X_star.shape[0]), X_star
                )

        s2 = s ** 2
        lambda_ = (sigma ** 2) / s2
        Cn = inv_penalized_cov(X, lam=lambda_)

        if return_cov == True:

            if X_star is None:

                beta_hat_ = np.dot(Cn, mo.crossprod(X, y))
                Sigma_hat_ = s2 * (
                    np.eye(X.shape[1])
                    - np.dot(Cn, mo.crossprod(X))
                )
                temp = np.dot(
                    Cn, mo.tcrossprod(Sigma_hat_, X)
                )
                smoothing_matrix = np.dot(X, temp)
                y_hat = np.dot(smoothing_matrix, y)

                return {
                    "beta_hat": beta_hat_,
                    "Sigma_hat": Sigma_hat_,
                    "GCV": np.mean(
                        (
                            (y - y_hat)
                            / (
                                1
                                - np.trace(smoothing_matrix)
                                / n
                            )
                        )
                        ** 2
                    ),
                }

            else:

                if beta_hat_ is None:
                    beta_hat_ = np.dot(
                        Cn, mo.crossprod(X, y)
                    )

                if Sigma_hat_ is None:
                    Sigma_hat_ = s2 * (
                        np.eye(X.shape[1])
                        - np.dot(Cn, mo.crossprod(X))
                    )

                temp = np.dot(
                    Cn, mo.tcrossprod(Sigma_hat_, X)
                )
                smoothing_matrix = np.dot(X, temp)
                y_hat = np.dot(smoothing_matrix, y)

                return {
                    "beta_hat": beta_hat_,
                    "Sigma_hat": Sigma_hat_,
                    "GCV": np.mean(
                        (
                            (y - y_hat)
                            / (
                                1
                                - np.trace(smoothing_matrix)
                                / n
                            )
                        )
                        ** 2
                    ),
                    "preds": np.dot(X_star, beta_hat_),
                    "preds_std": np.sqrt(
                        np.diag(
                            np.dot(
                                X_star,
                                mo.tcrossprod(
                                    Sigma_hat_, X_star
                                ),
                            )
                            + (sigma ** 2)
                            * np.eye(X_star.shape[0])
                        )
                    ),
                }

        else:  # return_cov == False

            if X_star is None:

                beta_hat_ = np.dot(Cn, mo.crossprod(X, y))
                Sigma_hat_ = s2 * (
                    np.eye(X.shape[1])
                    - np.dot(Cn, mo.crossprod(X))
                )
                temp = np.dot(
                    Cn, mo.tcrossprod(Sigma_hat_, X)
                )
                smoothing_matrix = np.dot(X, temp)
                y_hat = np.dot(smoothing_matrix, y)

                return {
                    "beta_hat": beta_hat_,
                    "GCV": np.mean(
                        (
                            (y - y_hat)
                            / (
                                1
                                - np.trace(smoothing_matrix)
                                / n
                            )
                        )
                        ** 2
                    ),
                }

            else:

                if beta_hat_ is None:
                    beta_hat_ = beta_hat(X, y, lam=lambda_)
                    Sigma_hat_ = s2 * (
                        np.eye(X.shape[1])
                        - np.dot(Cn, mo.crossprod(X))
                    )
                    temp = np.dot(
                        Cn, mo.tcrossprod(Sigma_hat_, X)
                    )
                    smoothing_matrix = np.dot(X, temp)
                    y_hat = np.dot(smoothing_matrix, y)

                return {
                    "beta_hat": beta_hat_,
                    "GCV": np.mean(
                        (
                            (y - y_hat)
                            / (
                                1
                                - np.trace(smoothing_matrix)
                                / n
                            )
                        )
                        ** 2
                    ),
                    "preds": np.dot(X_star, beta_hat_),
                }

    else:  # X is None | y is None  # predict

        assert (beta_hat_ is not None) & (
            X_star is not None
        )

        if return_cov == True:

            assert Sigma_hat_ is not None

            return {
                "preds": np.dot(X_star, beta_hat_),
                "preds_std": np.sqrt(
                    np.diag(
                        np.dot(
                            X_star,
                            mo.tcrossprod(
                                Sigma_hat_, X_star
                            ),
                        )
                        + (sigma ** 2)
                        * np.eye(X_star.shape[0])
                    )
                ),
            }

        else:

            return {"preds": np.dot(X_star, beta_hat_)}


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
            Cn = la.inv(
                np.dot(Sigma, mo.crossprod(X))
                + (sigma ** 2) * np.eye(p + 1)
            )

            if X_star is not None:
                X_star = mo.cbind(
                    np.ones(X_star.shape[0]), X_star
                )
        else:
            # rename to invCn
            Cn = la.inv(
                np.dot(Sigma, mo.crossprod(X))
                + (sigma ** 2) * np.eye(p)
            )

        temp = np.dot(Cn, mo.tcrossprod(Sigma, X))
        smoothing_matrix = np.dot(X, temp)
        y_hat = np.dot(smoothing_matrix, y)

        if return_cov == True:

            if X_star is None:

                return {
                    "beta_hat": np.dot(temp, y),
                    "Sigma_hat": Sigma
                    - np.dot(temp, np.dot(X, Sigma)),
                    "GCV": np.mean(
                        (
                            (y - y_hat)
                            / (
                                1
                                - np.trace(smoothing_matrix)
                                / n
                            )
                        )
                        ** 2
                    ),
                }

            else:

                if beta_hat_ is None:
                    beta_hat_ = np.dot(temp, y)

                if Sigma_hat_ is None:
                    Sigma_hat_ = Sigma - np.dot(
                        temp, np.dot(X, Sigma)
                    )

                return {
                    "beta_hat": beta_hat_,
                    "Sigma_hat": Sigma_hat_,
                    "GCV": np.mean(
                        (
                            (y - y_hat)
                            / (
                                1
                                - np.trace(smoothing_matrix)
                                / n
                            )
                        )
                        ** 2
                    ),
                    "preds": np.dot(X_star, beta_hat_),
                    "preds_std": np.sqrt(
                        np.diag(
                            np.dot(
                                X_star,
                                mo.tcrossprod(
                                    Sigma_hat_, X_star
                                ),
                            )
                            + (sigma ** 2)
                            * np.eye(X_star.shape[0])
                        )
                    ),
                }

        else:  # return_cov == False

            if X_star is None:

                return {
                    "beta_hat": np.dot(temp, y),
                    "GCV": np.mean(
                        (
                            (y - y_hat)
                            / (
                                1
                                - np.trace(smoothing_matrix)
                                / n
                            )
                        )
                        ** 2
                    ),
                }

            else:

                if beta_hat_ is None:
                    beta_hat_ = np.dot(temp, y)

                return {
                    "beta_hat": beta_hat_,
                    "GCV": np.mean(
                        (
                            (y - y_hat)
                            / (
                                1
                                - np.trace(smoothing_matrix)
                                / n
                            )
                        )
                        ** 2
                    ),
                    "preds": np.dot(X_star, beta_hat_),
                }

    else:  # (X is None) | (y is None) # predict

        assert (beta_hat_ is not None) & (
            X_star is not None
        )

        if return_cov == True:

            assert Sigma_hat_ is not None

            return {
                "preds": np.dot(X_star, beta_hat_),
                "preds_std": np.sqrt(
                    np.diag(
                        np.dot(
                            X_star,
                            mo.tcrossprod(
                                Sigma_hat_, X_star
                            ),
                        )
                        + (sigma ** 2)
                        * np.eye(X_star.shape[0])
                    )
                ),
            }

        else:

            return {"preds": np.dot(X_star, beta_hat_)}
