"""Tools for MTS."""

# Authors: Thierry Moudiki <thierry.moudiki@gmail.com>
#
# License: BSD 3 Clear

import numpy as np
import pandas as pd
try: 
    from sklearn.metrics import mean_pinball_loss
except ImportError:
    pass



# (block) bootstrap
def bootstrap(x, h, block_size=None, seed=123):
    """
    Generates block bootstrap indices for a given time series.

    Parameters:
    - x: numpy array, the original time series (univariate or multivariate).
    - h: int, output length
    - block_size: int, the size of the blocks to resample (if None, independent bootstrap).
    - seed: int, reproducibility seed.

    Returns:
    - numpy arrays containing resampled time series.
    """
    if len(x.shape) == 1:
        time_series_length = len(x)
        ndim = 1
    else:
        time_series_length = x.shape[0]
        ndim = x.shape[1]

    if block_size is not None:

        num_blocks = (time_series_length + block_size - 1) // block_size
        all_indices = np.arange(time_series_length)

        indices = []
        for i in range(num_blocks):
            np.random.seed(seed + i * 100)
            start_index = np.random.randint(
                0, time_series_length - block_size + 1
            )
            block_indices = all_indices[start_index: start_index + block_size]
            indices.extend(block_indices)

    else:  # block_size is None

        indices = np.random.choice(
            range(time_series_length), size=h, replace=True
        )

    if ndim == 1:
        return x[np.array(indices[:h])]
    else:
        return x[np.array(indices[:h]), :]


# compute input dates from data frame's index
def compute_input_dates(df):
    input_dates = df.index.values

    frequency = pd.infer_freq(pd.DatetimeIndex(input_dates))

    input_dates = pd.date_range(
        start=input_dates[0], periods=len(input_dates), freq=frequency
    ).values.tolist()

    df_input_dates = pd.DataFrame({"date": input_dates})

    return pd.to_datetime(df_input_dates["date"]).dt.date


# compute output dates from data frame's index
def compute_output_dates(df, horizon):
    input_dates = df.index.values

    if input_dates[0] == 0:
        input_dates = pd.date_range(
            start=pd.Timestamp.today().strftime("%Y-%m-%d"), periods=horizon
        )

    # print(f"\n in nnetsauce.utils.timeseries 1: {input_dates} \n")

    frequency = pd.infer_freq(pd.DatetimeIndex(input_dates))

    # print(f"\n in nnetsauce.utils.timeseries 2: {frequency} \n")

    output_dates = np.delete(
        pd.date_range(
            start=input_dates[-1], periods=horizon + 1, freq=frequency
        ).values,
        0,
    ).tolist()

    # print(f"\n in nnetsauce.utils.timeseries 3: {output_dates} \n")

    df_output_dates = pd.DataFrame({"date": output_dates})
    output_dates = pd.to_datetime(df_output_dates["date"]).dt.date

    return output_dates, frequency


def coverage(obj, actual, level=95, per_series=False):

    alpha = 1 - level / 100
    lt = obj.lower
    ut = obj.upper
    n_points = actual.shape[0]

    if isinstance(lt, pd.DataFrame):
        actual = actual.values

    # Ensure the arrays have the same length
    assert (
        n_points == lt.shape[0] == ut.shape[0]
    ), "actual, lower and upper bounds have different shapes"

    if isinstance(lt, pd.DataFrame) and isinstance(ut, pd.DataFrame):
        diff_lt = (lt.values <= actual) * (ut.values >= actual)
    else:
        diff_lt = (lt <= actual) * (ut >= actual)

    if per_series:
        return np.mean(diff_lt, axis=0) * 100
    else:
        return np.mean(diff_lt) * 100


# create lags for one series
def create_lags(x, k, n=None):
    k_ = k + 1

    n_k = len(x) - k_

    x_ = x[::-1]

    z = [x_[i: (n_k + i + 1)] for i in range(k_)]

    if n is None:
        return np.column_stack(z)

    temp = np.column_stack(z)
    return np.repeat(temp, n).reshape(temp.shape[0], -1)


# create inputs for training from MTS (envisage other regressors in X)
# X in decreasing order (!)
# a = np.reshape(range(0, 24), (8, 3))
# create_train_inputs(a, 2)
def create_train_inputs(X, k):
    n_k = X.shape[0] - k

    z = [X[i: n_k + i, :] for i in range(1, (k + 1))]

    return (X[0:n_k, :], np.column_stack(z))


# reformat response in prediction loop
# a = np.reshape(range(0, 24), (8, 3))
# reformat_response(a, 2)
def reformat_response(X, k):
    return np.concatenate([X[i, :] for i in range(k)])


def winkler_score(obj, actual, level=95, per_series=False):

    alpha = 1 - level / 100
    lt = obj.lower
    ut = obj.upper
    n_points = actual.shape[0]

    # Ensure the arrays have the same length
    assert (
        n_points == lt.shape[0] == ut.shape[0]
    ), "actual, lower and upper bounds have different shapes"

    if isinstance(lt, pd.DataFrame) and isinstance(actual, pd.DataFrame):
        diff_lt = lt.values - actual.values
    else:
        diff_lt = lt - actual
    if isinstance(lt, pd.DataFrame) and isinstance(ut, pd.DataFrame):
        diff_bounds = ut.values - lt.values
    else:
        diff_bounds = ut - lt
    if isinstance(lt, pd.DataFrame) and isinstance(ut, pd.DataFrame):
        diff_ut = actual.values - ut.values
    else:
        diff_ut = actual - ut

    score = diff_bounds + (2 / alpha) * (
        np.maximum(diff_lt, 0) + np.maximum(diff_ut, 0)
    )

    if per_series == True:
        return np.mean(score, axis=0)
    else:
        return np.mean(score)


def mean_errors(
    pred, actual, scoring="root_mean_squared_error", per_series=False
):

    if isinstance(pred, pd.DataFrame):
        pred = pred.values
    else:
        pred = pred.mean.values

    if isinstance(actual, pd.DataFrame):
        actual = actual.values

    diff = pred - actual

    if per_series == True:

        if scoring == "mean_error":
            return np.mean(diff, axis=0).tolist()
        elif scoring == "mean_absolute_error":
            return np.mean(np.abs(diff), axis=0).tolist()
        elif scoring == "mean_percentage_error":
            return np.asarray(np.mean(diff / actual, axis=0) * 100).tolist()
        elif scoring == "mean_absolute_percentage_error":
            return np.asarray(
                np.mean(np.abs(diff / actual), axis=0) * 100
            ).tolist()
        elif scoring == "root_mean_squared_error":
            return np.sqrt(np.mean(np.square(diff), axis=0)).tolist()
        elif scoring == "mean_squared_error":
            return np.mean(np.square(diff), axis=0).tolist()
        elif scoring == "mean_pinball_loss":
            return [
                mean_pinball_loss(actual[:, i], pred[:, i])
                for i in range(actual.shape[1])
            ]

    else:

        if scoring == "mean_error":
            return np.mean(diff)
        elif scoring == "mean_absolute_error":
            return np.mean(np.abs(diff))
        elif scoring == "mean_percentage_error":
            return np.mean(diff / actual) * 100
        elif scoring == "mean_absolute_percentage_error":
            return np.mean(np.abs(diff / actual)) * 100
        elif scoring == "root_mean_squared_error":
            return np.sqrt(np.mean(np.square(diff)))
        elif scoring == "mean_squared_error":
            return np.mean(np.square(diff))
        elif scoring == "mean_pinball_loss":
            return mean_pinball_loss(actual, pred)


# create k lags for series x
# def create_lags(x, k):
#
#    n = len(x)
#    n_k = n - k
#    n_col = k + 1
#    res = np.zeros((n_k, n_col))
#
#    for i in range(n_k):
#        for j in range(n_col):
#            res[i, j] = x[i + j]
#
#    return res


# create inputs for training from MTS (envisage other regressors in X)
# X in decreasing order (!)
# def create_train_inputs(X, k):
#
#    n, p = X.shape
#    n_k = n - k
#
#    y = np.zeros((n_k, p))
#    xreg = np.zeros((n_k, k*p))
#
#    for j in range(p):
#
#        Y_x = create_lags(X[:,j], k)
#
#        y[:,j] = Y_x[:, 0]
#
#        for l in range(1, k):
#
#            xreg[:, j*k + (l-1)] = Y_x[:, l]
#
#        xreg[:, j*k + (k-1)] = Y_x[:, k]
#
#    return (y, xreg)


# reformat response in prediction loop
# def reformat_response(X, k):
#
#    n, p = X.shape
#    res = np.zeros((1, k*p))
#
#    for j in range(p):
#        for i in range(k):
#            res[0, j*k + i] = X[i, j]
#
#    return res
#
