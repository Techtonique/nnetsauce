import numpy as np

from .memoize import memoize
from scipy import sparse
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
)
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture


# column bind
def cbind(x, y):

    # if len(x.shape) == 1 or len(y.shape) == 1:
    return np.column_stack((x, y))


# center... response
@memoize
def center_response(y):
    y_mean = np.mean(y)
    return y_mean, (y - y_mean)


# cluster the covariates
def cluster_covariates(
    X, n_clusters, seed, type_clust="kmeans", **kwargs
):

    if type_clust == "kmeans":

        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=seed,
            **kwargs
        )
        kmeans.fit(X)

        return kmeans, kmeans.predict(X)

    if type_clust == "gmm":

        gmm = GaussianMixture(
            n_components=n_clusters,
            random_state=seed,
            **kwargs
        )
        gmm.fit(X)

        return gmm, gmm.predict(X)


# computes t(x)%*%y
def crossprod(x, y=None):
    # assert on dimensions
    if y is None:
        return np.dot(x.transpose(), x)
    else:
        return np.dot(x.transpose(), y)


# dropout
def dropout(x, drop_prob=0, seed=123):

    assert 0 <= drop_prob <= 1

    n, p = x.shape

    if drop_prob == 0:
        return x

    if drop_prob == 1:
        return np.zeros_like(x)

    np.random.seed(seed)
    dropped_indices = np.random.rand(n, p) > drop_prob

    return dropped_indices * x / (1 - drop_prob)


# one-hot encoding
@memoize
def one_hot_encode(x_clusters, n_clusters):

    assert (
        max(x_clusters) <= n_clusters
    ), "you must have max(x_clusters) <= n_clusters"

    n_obs = len(x_clusters)
    res = np.zeros((n_obs, n_clusters))

    for i in range(n_obs):
        res[i, x_clusters[i]] = 1

    return res


# one-hot encoding
@memoize
def one_hot_encode2(y, n_classes):
    
    n_obs = len(y)
    
    res = np.zeros((n_obs, n_classes))

    for i in range(n_obs):
        res[i, y[i]] = 1

    return res


# row bind
def rbind(x, y):

    # if len(x.shape) == 1 or len(y.shape) == 1:
    return np.row_stack((x, y))


# from sklearn.utils.exmath
def safe_sparse_dot(a, b, dense_output=False):
    """Dot product that handle the sparse matrix case correctly

    Uses BLAS GEMM as replacement for numpy.dot where possible
    to avoid unnecessary copies.

    Parameters
    ----------
    a : array or sparse matrix
    b : array or sparse matrix
    dense_output : boolean, default False
        When False, either ``a`` or ``b`` being sparse will yield sparse
        output. When True, output will always be an array.

    Returns
    -------
    dot_product : array or sparse matrix
        sparse if ``a`` or ``b`` is sparse and ``dense_output=False``.
    """
    if sparse.issparse(a) or sparse.issparse(b):
        ret = a * b
        if dense_output and hasattr(ret, "toarray"):
            ret = ret.toarray()
        return ret
    else:
        return np.dot(a, b)


# scale... covariates
def scale_covariates(
    X, choice="std", training=True, scaler=None
):

    scaling_options = {
        "std": StandardScaler(
            copy=True, with_mean=True, with_std=True
        ),
        "minmax": MinMaxScaler(),
    }

    if training == True:
        # scaler must be not None
        scaler = scaling_options[choice]
        scaled_X = scaler.fit_transform(X)
        return scaler, scaled_X

    # training == False:
    # scaler must be not None
    return scaler.transform(X)


# from sklearn.utils.exmath
def squared_norm(x):
    """Squared Euclidean or Frobenius norm of x.

    Faster than norm(x) ** 2.

    Parameters
    ----------
    x : array_like

    Returns
    -------
    float
        The Euclidean norm when x is a vector, the Frobenius norm when x
        is a matrix (2-d array).
    """
    x = np.ravel(x, order="K")
    return np.dot(x, x)


# computes x%*%t(y)
def tcrossprod(x, y=None):
    # assert on dimensions
    if y is None:
        return np.dot(x, x.transpose())
    else:
        return np.dot(x, y.transpose())


# convert vector to numpy array
def to_np_array(X):
    return np.array(X.copy(), ndmin=2)


# scale matrix
# def scale_matrix(X, x_means=None, x_vars=None):
#
#    if ((x_means is None) & (x_vars is None)):
#        x_means = X.mean(axis = 0)
#        x_vars = X.var(axis = 0)
#        return ((X - x_means)/np.sqrt(x_vars),
#                x_means,
#                x_vars)
#
#    if ((x_means is not None) & (x_vars is None)):
#        return X - x_means
#
#    if ((x_means is None) & (x_vars is not None)):
#        return X/np.sqrt(x_vars)
#
#    if ((x_means is not None) & (x_vars is not None)):
#        return (X - x_means)/np.sqrt(x_vars)
