#!/usr/bin/env python

"""
Nonconformity functions.
"""

# Authors: Henrik Linusson
# Yaniv Romano modified RegressorNc class to include CQR

from __future__ import division

import abc
import numpy as np
import sklearn.base
from .base import ClassifierAdapter, RegressorAdapter
from .base import OobClassifierAdapter, OobRegressorAdapter

# -----------------------------------------------------------------------------
# Error functions
# -----------------------------------------------------------------------------


class ClassificationErrFunc(object):
    """Base class for classification model error functions."""

    __metaclass__ = abc.ABCMeta

    def __init__(self):
        super(ClassificationErrFunc, self).__init__()

    @abc.abstractmethod
    def apply(self, prediction, y):
        """Apply the nonconformity function.

        Parameters
        ----------
        prediction : numpy array of shape [n_samples, n_classes]
            Class probability estimates for each sample.

        y : numpy array of shape [n_samples]
            True output labels of each sample.

        Returns
        -------
        nc : numpy array of shape [n_samples]
            Nonconformity scores of the samples.
        """
        pass


class RegressionErrFunc(object):
    """Base class for regression model error functions."""

    __metaclass__ = abc.ABCMeta

    def __init__(self):
        super(RegressionErrFunc, self).__init__()

    @abc.abstractmethod
    def apply(self, prediction, y):  # , norm=None, beta=0):
        """Apply the nonconformity function.

        Parameters
        ----------
        prediction : numpy array of shape [n_samples, n_classes]
            Class probability estimates for each sample.

        y : numpy array of shape [n_samples]
            True output labels of each sample.

        Returns
        -------
        nc : numpy array of shape [n_samples]
            Nonconformity scores of the samples.
        """
        pass

    @abc.abstractmethod
    def apply_inverse(self, nc, significance):  # , norm=None, beta=0):
        """Apply the inverse of the nonconformity function (i.e.,
        calculate prediction interval).

        Parameters
        ----------
        nc : numpy array of shape [n_calibration_samples]
            Nonconformity scores obtained for conformal predictor.

        significance : float
            Significance level (0, 1).

        Returns
        -------
        interval : numpy array of shape [n_samples, 2]
            Minimum and maximum interval boundaries for each prediction.
        """
        pass


class InverseProbabilityErrFunc(ClassificationErrFunc):
    """Calculates the probability of not predicting the correct class."""

    def __init__(self):
        super(InverseProbabilityErrFunc, self).__init__()

    def apply(self, prediction, y):
        prob = np.zeros(y.size, dtype=np.float32)
        for i, y_ in enumerate(y):
            if y_ >= prediction.shape[1]:
                prob[i] = 0
            else:
                prob[i] = prediction[i, int(y_)]
        return 1 - prob


class MarginErrFunc(ClassificationErrFunc):
    """
    Calculates the margin error.
    """

    def __init__(self):
        super(MarginErrFunc, self).__init__()

    def apply(self, prediction, y):
        prob = np.zeros(y.size, dtype=np.float32)
        for i, y_ in enumerate(y):
            if y_ >= prediction.shape[1]:
                prob[i] = 0
            else:
                prob[i] = prediction[i, int(y_)]
                prediction[i, int(y_)] = -np.inf
        return 0.5 - ((prob - prediction.max(axis=1)) / 2)


class AbsErrorErrFunc(RegressionErrFunc):
    """Calculates absolute error nonconformity for regression problems."""

    def __init__(self):
        super(AbsErrorErrFunc, self).__init__()

    def apply(self, prediction, y):
        return np.abs(prediction - y)

    def apply_inverse(self, nc, significance):
        nc = np.sort(nc)[::-1]
        border = int(np.floor(significance * (nc.size + 1))) - 1
        # TODO: should probably warn against too few calibration examples
        border = min(max(border, 0), nc.size - 1)
        return np.vstack([nc[border], nc[border]])


class SignErrorErrFunc(RegressionErrFunc):
    """Calculates signed error nonconformity for regression problems."""

    def __init__(self):
        super(SignErrorErrFunc, self).__init__()

    def apply(self, prediction, y):
        return prediction - y

    def apply_inverse(self, nc, significance):

        err_high = -nc
        err_low = nc

        err_high = np.reshape(err_high, (nc.shape[0], 1))
        err_low = np.reshape(err_low, (nc.shape[0], 1))

        nc = np.concatenate((err_low, err_high), 1)

        nc = np.sort(nc, 0)
        index = int(np.ceil((1 - significance / 2) * (nc.shape[0] + 1))) - 1
        index = min(max(index, 0), nc.shape[0] - 1)
        return np.vstack([nc[index, 0], nc[index, 1]])


# CQR symmetric error function
class QuantileRegErrFunc(RegressionErrFunc):
    """Calculates conformalized quantile regression error."""

    def __init__(self):
        super(QuantileRegErrFunc, self).__init__()

    def apply(self, prediction, y):
        y_lower = prediction[:, 0]
        y_upper = prediction[:, -1]
        error_low = y_lower - y
        error_high = y - y_upper
        err = np.maximum(error_high, error_low)
        return err

    def apply_inverse(self, nc, significance):
        nc = np.sort(nc, 0)
        index = int(np.ceil((1 - significance) * (nc.shape[0] + 1))) - 1
        index = min(max(index, 0), nc.shape[0] - 1)
        return np.vstack([nc[index], nc[index]])


# CQR asymmetric error function
class QuantileRegAsymmetricErrFunc(RegressionErrFunc):
    """Calculates conformalized quantile regression asymmetric error function."""

    def __init__(self):
        super(QuantileRegAsymmetricErrFunc, self).__init__()

    def apply(self, prediction, y):
        y_lower = prediction[:, 0]
        y_upper = prediction[:, -1]

        error_high = y - y_upper
        error_low = y_lower - y

        err_high = np.reshape(error_high, (y_upper.shape[0], 1))
        err_low = np.reshape(error_low, (y_lower.shape[0], 1))

        return np.concatenate((err_low, err_high), 1)

    def apply_inverse(self, nc, significance):
        nc = np.sort(nc, 0)
        index = int(np.ceil((1 - significance / 2) * (nc.shape[0] + 1))) - 1
        index = min(max(index, 0), nc.shape[0] - 1)
        return np.vstack([nc[index, 0], nc[index, 1]])


# -----------------------------------------------------------------------------
# Base nonconformity scorer
# -----------------------------------------------------------------------------
class BaseScorer(sklearn.base.BaseEstimator):
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        super(BaseScorer, self).__init__()

    @abc.abstractmethod
    def fit(self, x, y):
        pass

    @abc.abstractmethod
    def score(self, x, y=None):
        pass


class RegressorNormalizer(BaseScorer):
    def __init__(self, base_model, normalizer_model, err_func):
        super(RegressorNormalizer, self).__init__()
        self.base_model = base_model
        self.normalizer_model = normalizer_model
        self.err_func = err_func

    def fit(self, x, y):
        residual_prediction = self.base_model.predict(x)
        residual_error = np.abs(self.err_func.apply(residual_prediction, y))

        ######################################################################
        # Optional: use logarithmic function as in the original implementation
        # available in https://github.com/donlnz/nonconformist
        #
        # CODE:
        # residual_error += 0.00001 # Add small term to avoid log(0)
        # log_err = np.log(residual_error)
        ######################################################################

        log_err = residual_error
        self.normalizer_model.fit(x, log_err)

    def score(self, x, y=None):

        ######################################################################
        # Optional: use logarithmic function as in the original implementation
        # available in https://github.com/donlnz/nonconformist
        #
        # CODE:
        # norm = np.exp(self.normalizer_model.predict(x))
        ######################################################################

        norm = np.abs(self.normalizer_model.predict(x))
        return norm


class NcFactory(object):
    @staticmethod
    def create_nc(model, err_func=None, normalizer_model=None, oob=False):
        if normalizer_model is not None:
            normalizer_adapter = RegressorAdapter(normalizer_model)
        else:
            normalizer_adapter = None

        if isinstance(model, sklearn.base.ClassifierMixin):
            err_func = MarginErrFunc() if err_func is None else err_func
            if oob:
                c = sklearn.base.clone(model)
                c.fit([[0], [1]], [0, 1])
                if hasattr(c, "oob_decision_function_"):
                    adapter = OobClassifierAdapter(model)
                else:
                    raise AttributeError(
                        "Cannot use out-of-bag "
                        "calibration with {}".format(model.__class__.__name__)
                    )
            else:
                adapter = ClassifierAdapter(model)

            if normalizer_adapter is not None:
                normalizer = RegressorNormalizer(adapter, normalizer_adapter, err_func)
                return ClassifierNc(adapter, err_func, normalizer)
            else:
                return ClassifierNc(adapter, err_func)

        elif isinstance(model, sklearn.base.RegressorMixin):
            err_func = AbsErrorErrFunc() if err_func is None else err_func
            if oob:
                c = sklearn.base.clone(model)
                c.fit([[0], [1]], [0, 1])
                if hasattr(c, "oob_prediction_"):
                    adapter = OobRegressorAdapter(model)
                else:
                    raise AttributeError(
                        "Cannot use out-of-bag "
                        "calibration with {}".format(model.__class__.__name__)
                    )
            else:
                adapter = RegressorAdapter(model)

            if normalizer_adapter is not None:
                normalizer = RegressorNormalizer(adapter, normalizer_adapter, err_func)
                return RegressorNc(adapter, err_func, normalizer)
            else:
                return RegressorNc(adapter, err_func)


class BaseModelNc(BaseScorer):
    """Base class for nonconformity scorers based on an underlying model.

    Parameters
    ----------
    model : ClassifierAdapter or RegressorAdapter
        Underlying classification model used for calculating nonconformity
        scores.

    err_func : ClassificationErrFunc or RegressionErrFunc
        Error function object.

    normalizer : BaseScorer
        Normalization model.

    beta : float
        Normalization smoothing parameter. As the beta-value increases,
        the normalized nonconformity function approaches a non-normalized
        equivalent.
    """

    def __init__(self, model, err_func, normalizer=None, beta=1e-6):
        super(BaseModelNc, self).__init__()
        self.err_func = err_func
        self.model = model
        self.normalizer = normalizer
        self.beta = beta

        # If we use sklearn.base.clone (e.g., during cross-validation),
        # object references get jumbled, so we need to make sure that the
        # normalizer has a reference to the proper model adapter, if applicable.
        if self.normalizer is not None and hasattr(self.normalizer, "base_model"):
            self.normalizer.base_model = self.model

        self.last_x, self.last_y = None, None
        self.last_prediction = None
        self.clean = False

    def fit(self, x, y):
        """Fits the underlying model of the nonconformity scorer.

        Parameters
        ----------
        x : numpy array of shape [n_samples, n_features]
            Inputs of examples for fitting the underlying model.

        y : numpy array of shape [n_samples]
            Outputs of examples for fitting the underlying model.

        Returns
        -------
        None
        """
        self.model.fit(x, y)
        if self.normalizer is not None:
            self.normalizer.fit(x, y)
        self.clean = False

    def score(self, x, y=None):
        """Calculates the nonconformity score of a set of samples.

        Parameters
        ----------
        x : numpy array of shape [n_samples, n_features]
            Inputs of examples for which to calculate a nonconformity score.

        y : numpy array of shape [n_samples]
            Outputs of examples for which to calculate a nonconformity score.

        Returns
        -------
        nc : numpy array of shape [n_samples]
            Nonconformity scores of samples.
        """
        prediction = self.model.predict(x)
        n_test = x.shape[0]
        if self.normalizer is not None:
            norm = self.normalizer.score(x) + self.beta
        else:
            norm = np.ones(n_test)
        if prediction.ndim > 1:
            ret_val = self.err_func.apply(prediction, y)
        else:
            ret_val = self.err_func.apply(prediction, y) / norm
        return ret_val


# -----------------------------------------------------------------------------
# Classification nonconformity scorers
# -----------------------------------------------------------------------------
class ClassifierNc(BaseModelNc):
    """Nonconformity scorer using an underlying class probability estimating
    model.

    Parameters
    ----------
    model : ClassifierAdapter
        Underlying classification model used for calculating nonconformity
        scores.

    err_func : ClassificationErrFunc
        Error function object.

    normalizer : BaseScorer
        Normalization model.

    beta : float
        Normalization smoothing parameter. As the beta-value increases,
        the normalized nonconformity function approaches a non-normalized
        equivalent.

    Attributes
    ----------
    model : ClassifierAdapter
        Underlying model object.

    err_func : ClassificationErrFunc
        Scorer function used to calculate nonconformity scores.

    See also
    --------
    RegressorNc, NormalizedRegressorNc
    """

    def __init__(self, model, err_func=MarginErrFunc(), normalizer=None, beta=1e-6):
        super(ClassifierNc, self).__init__(model, err_func, normalizer, beta)


# -----------------------------------------------------------------------------
# Regression nonconformity scorers
# -----------------------------------------------------------------------------
class RegressorNc(BaseModelNc):
    """Nonconformity scorer using an underlying regression model.

    Parameters
    ----------
    model : RegressorAdapter
        Underlying regression model used for calculating nonconformity scores.

    err_func : RegressionErrFunc
        Error function object.

    normalizer : BaseScorer
        Normalization model.

    beta : float
        Normalization smoothing parameter. As the beta-value increases,
        the normalized nonconformity function approaches a non-normalized
        equivalent.

    Attributes
    ----------
    model : RegressorAdapter
        Underlying model object.

    err_func : RegressionErrFunc
        Scorer function used to calculate nonconformity scores.

    See also
    --------
    ProbEstClassifierNc, NormalizedRegressorNc
    """

    def __init__(self, model, err_func=AbsErrorErrFunc(), normalizer=None, beta=1e-6):
        super(RegressorNc, self).__init__(model, err_func, normalizer, beta)

    def predict(self, x, nc, significance=None):
        """Constructs prediction intervals for a set of test examples.

        Predicts the output of each test pattern using the underlying model,
        and applies the (partial) inverse nonconformity function to each
        prediction, resulting in a prediction interval for each test pattern.

        Parameters
        ----------
        x : numpy array of shape [n_samples, n_features]
            Inputs of patters for which to predict output values.

        significance : float
            Significance level (maximum allowed error rate) of predictions.
            Should be a float between 0 and 1. If ``None``, then intervals for
            all significance levels (0.01, 0.02, ..., 0.99) are output in a
            3d-matrix.

        Returns
        -------
        p : numpy array of shape [n_samples, 2] or [n_samples, 2, 99]
            If significance is ``None``, then p contains the interval (minimum
            and maximum boundaries) for each test pattern, and each significance
            level (0.01, 0.02, ..., 0.99). If significance is a float between
            0 and 1, then p contains the prediction intervals (minimum and
            maximum	boundaries) for the set of test patterns at the chosen
            significance level.
        """
        n_test = x.shape[0]
        prediction = self.model.predict(x)
        if self.normalizer is not None:
            norm = self.normalizer.score(x) + self.beta
        else:
            norm = np.ones(n_test)

        if significance:
            intervals = np.zeros((x.shape[0], 2))
            err_dist = self.err_func.apply_inverse(nc, significance)
            err_dist = np.hstack([err_dist] * n_test)
            if prediction.ndim > 1:  # CQR
                intervals[:, 0] = prediction[:, 0] - err_dist[0, :]
                intervals[:, 1] = prediction[:, -1] + err_dist[1, :]
            else:  # regular conformal prediction
                err_dist *= norm
                intervals[:, 0] = prediction - err_dist[0, :]
                intervals[:, 1] = prediction + err_dist[1, :]

            return intervals
        else:  # Not tested for CQR
            significance = np.arange(0.01, 1.0, 0.01)
            intervals = np.zeros((x.shape[0], 2, significance.size))

            for i, s in enumerate(significance):
                err_dist = self.err_func.apply_inverse(nc, s)
                err_dist = np.hstack([err_dist] * n_test)
                err_dist *= norm

                intervals[:, 0, i] = prediction - err_dist[0, :]
                intervals[:, 1, i] = prediction + err_dist[0, :]

            return intervals
