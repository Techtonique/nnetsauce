"""Model selection"""

# Authors: Thierry Moudiki
#
# License: BSD 3

# MTS -----

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from scipy.stats import describe
from tqdm import tqdm
from ..utils.timeseries import coverage, mean_errors, winkler_score


class TimeSeriesSplit(TimeSeriesSplit):
    """Time Series cross-validator"""

    def __init__(self, n_splits=5, max_train_size=None):
        super().__init__(n_splits=n_splits, max_train_size=max_train_size)

    def split(
        self,
        X,
        y=None,
        groups=None,
        initial_window=5,
        horizon=3,
        fixed_window=False,
    ):
        """Generate indices to split data into training and test set.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.

        initial_window : int, initial number of consecutive values in each
                         training set sample

        horizon : int, number of consecutive values in test set sample

        fixed_window : boolean, if False, all training samples start at index 0

        y : array-like, shape (n_samples,)
            Always ignored, exists for compatibility.

        groups : array-like, with shape (n_samples,)
            Always ignored, exists for compatibility.

        Yields
        ------
        train : ndarray
            The training set indices for that split.

        test : ndarray
            The testing set indices for that split.
        """
        # assert initial_window
        # assert horizon
        # assert fixed_window

        try:
            n = X.shape[0]
        except:
            n = len(X)

        # Initialization of indices -----

        indices = np.arange(n)
        n_splits = 0

        # train index
        min_index_train = 0
        max_index_train = initial_window

        # test index
        min_index_test = max_index_train
        max_index_test = initial_window + horizon

        # Main loop -----

        if fixed_window == True:
            while max_index_test <= n:
                yield (
                    indices[min_index_train:max_index_train],
                    indices[min_index_test:max_index_test],
                )

                min_index_train += 1
                min_index_test += 1
                max_index_train += 1
                max_index_test += 1

                n_splits += 1

        else:
            while max_index_test <= n:
                yield (
                    indices[min_index_train:max_index_train],
                    indices[min_index_test:max_index_test],
                )

                max_index_train += 1
                min_index_test += 1
                max_index_test += 1

                n_splits += 1

        # set n_splits after (?)
        self.n_splits = n_splits
        self.max_train_size = max_index_train + 1


def cross_val_score(
    estimator,
    X,
    scoring="root_mean_squared_error",
    n_jobs=None,
    verbose=0,
    xreg=None,
    initial_window=5,
    horizon=3,
    fixed_window=False,
    show_progress=True,
    level=95,
    **kwargs,
):
    """Evaluate a score by time series cross-validation.

    Parameters:

        estimator: estimator object implementing `fit`, and of class `MTS`
            The object to use to fit the data.

        X: {array-like, sparse matrix} of shape (n_samples, n_features)
            The data to fit.

        scoring: str or a function
            A str in ('root_mean_squared_error', 'mean_squared_error', 'mean_error',
            'mean_absolute_error', 'mean_error', 'mean_percentage_error',
            'mean_absolute_percentage_error',  'winkler_score', 'coverage')
            Or a function defined as 'coverage' and 'winkler_score' in `utils.timeseries`

        n_jobs: int, default=None
            Number of jobs to run in parallel.

        verbose: int, default=0
            The verbosity level.

        xreg: array-like, optional (default=None)
            Additional (external) regressors to be passed to estimator.obj
            xreg must be in 'increasing' order (most recent observations last)

        initial_window: int
            initial number of consecutive values in each training set sample

        horizon: int
            number of consecutive values in test set sample

        fixed_window: boolean
            if False, all training samples start at index 0, and the training
            window's size is increasing.
            if True, the training window's size is fixed, and the window is
            rolling forward

        show_progress: boolean
            if True, a progress bar is printed

        **kwargs: dict
            additional parameters to be passed to `estimator`'s `fit` and `predict`

    Returns:

        A tuple: descriptive statistics or errors and raw errors

    """
    tscv = TimeSeriesSplit()

    tscv_obj = tscv.split(
        X,
        initial_window=initial_window,
        horizon=horizon,
        fixed_window=fixed_window,
    )

    if isinstance(scoring, str):

        assert scoring in (
            "root_mean_squared_error",
            "mean_squared_error",
            "mean_error",
            "mean_absolute_error",
            "mean_percentage_error",
            "mean_absolute_percentage_error",
            "winkler_score",
            "coverage",
        ), "must have scoring in ('root_mean_squared_error', 'mean_squared_error', 'mean_error', 'mean_absolute_error', 'mean_error', 'mean_percentage_error', 'mean_absolute_percentage_error',  'winkler_score', 'coverage')"

        def err_func(X_test, X_pred, scoring):
            if (estimator.replications is not None) or (
                estimator.type_pi == "gaussian"
            ):  # probabilistic
                if scoring == "winkler_score":
                    return winkler_score(X_pred, X_test, level=level)
                elif scoring == "coverage":
                    return coverage(X_pred, X_test, level=level)
                else:
                    return mean_errors(
                        pred=X_pred.mean, actual=X_test, scoring=scoring
                    )
            else:  # not probabilistic
                return mean_errors(pred=X_pred, actual=X_test, scoring=scoring)

    else:  # isinstance(scoring, str) = False

        err_func = scoring

    errors = []

    if show_progress is True:
        iterator = tqdm(tscv_obj, total=tscv.n_splits)
    else:
        iterator = tscv_obj

    for train_index, test_index in iterator:

        if verbose == 1:
            print(f"TRAIN: {train_index}")
            print(f"TEST: {test_index}")

        if isinstance(X, pd.DataFrame):
            estimator.fit(X.iloc[train_index, :], xreg=xreg, **kwargs)
            X_test = X.iloc[test_index, :]
        else:
            estimator.fit(X[train_index, :], xreg=xreg, **kwargs)
            X_test = X[test_index, :]
        X_pred = estimator.predict(
            h=int(len(test_index)), level=level, **kwargs
        )

        errors.append(err_func(X_test, X_pred, scoring))

    res = np.asarray(errors)

    return res, describe(res)
