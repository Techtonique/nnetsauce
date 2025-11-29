import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted
from sklearn.kernel_approximation import RBFSampler


def _get_estimator_type(estimator):
    """Determine if estimator is classifier or regressor."""
    if hasattr(estimator, "_estimator_type"):
        return estimator._estimator_type
    elif hasattr(estimator, "predict_proba") or hasattr(
        estimator, "decision_function"
    ):
        return "classifier"
    elif hasattr(estimator, "predict"):
        return "regressor"
    else:
        return None


class RandomFourierEstimator(BaseEstimator):
    def __init__(
        self, estimator, n_components=100, gamma=1.0, random_state=None
    ):
        """
        Random Fourier Features transformation with a given estimator.

        Parameters:
        - estimator: A scikit-learn estimator (classifier, regressor, etc.).
        - n_components: Number of random Fourier features.
        - gamma: Hyperparameter for RBF kernel approximation.
        - random_state: Random state for reproducibility.
        """
        self.estimator = estimator
        self.n_components = n_components
        self.gamma = gamma
        self.random_state = random_state

        # Dynamically set the estimator type and appropriate mixin
        estimator_type = _get_estimator_type(estimator)
        if estimator_type == "classifier":
            self._estimator_type = "classifier"
            # Add ClassifierMixin to the class hierarchy
            if not isinstance(self, ClassifierMixin):
                self.__class__ = type(
                    self.__class__.__name__,
                    (self.__class__, ClassifierMixin),
                    dict(self.__class__.__dict__),
                )
        elif estimator_type == "regressor":
            self._estimator_type = "regressor"
            # Add RegressorMixin to the class hierarchy
            if not isinstance(self, RegressorMixin):
                self.__class__ = type(
                    self.__class__.__name__,
                    (self.__class__, RegressorMixin),
                    dict(self.__class__.__dict__),
                )

    def fit(self, X, y=None):
        """
        Fit the Random Fourier feature transformer and the estimator.
        """
        X = check_array(X)

        # Initialize and fit the Random Fourier Feature transformer
        self.rff_ = RBFSampler(
            n_components=self.n_components,
            gamma=self.gamma,
            random_state=self.random_state,
        )
        X_transformed = self.rff_.fit_transform(X)

        # Fit the underlying estimator on the transformed data
        self.estimator.fit(X_transformed, y)

        return self

    def partial_fit(self, X, y, classes=None):
        """
        Incrementally fit the Random Fourier feature transformer and the estimator.
        """
        X = check_array(X)

        # Check if RFF transformer is already fitted
        if not hasattr(self, "rff_"):
            # First call - fit the transformer
            self.rff_ = RBFSampler(
                n_components=self.n_components,
                gamma=self.gamma,
                random_state=self.random_state,
            )
            X_transformed = self.rff_.fit_transform(X)
        else:
            # Subsequent calls - only transform
            X_transformed = self.rff_.transform(X)

        # If estimator supports partial_fit, we use it, otherwise raise an error
        if hasattr(self.estimator, "partial_fit"):
            self.estimator.partial_fit(X_transformed, y, classes=classes)
        else:
            raise ValueError(
                f"The estimator {type(self.estimator).__name__} does not support partial_fit method."
            )

        return self

    def predict(self, X):
        """
        Predict using the Random Fourier transformed data.
        """
        check_is_fitted(self, ["rff_"])
        X = check_array(X)

        # Transform the input data
        X_transformed = self.rff_.transform(X)

        # Predict using the underlying estimator
        return self.estimator.predict(X_transformed)

    def predict_proba(self, X):
        """
        Predict class probabilities (only for classifiers).
        """
        if (
            not hasattr(self, "_estimator_type")
            or self._estimator_type != "classifier"
        ):
            raise AttributeError(
                "predict_proba is not available for this estimator type."
            )

        check_is_fitted(self, ["rff_"])
        X = check_array(X)

        if not hasattr(self.estimator, "predict_proba"):
            raise ValueError(
                f"The estimator {type(self.estimator).__name__} does not support predict_proba."
            )

        # Transform the input data
        X_transformed = self.rff_.transform(X)

        # Predict probabilities using the underlying estimator
        return self.estimator.predict_proba(X_transformed)

    def predict_log_proba(self, X):
        """
        Predict class log probabilities (only for classifiers).
        """
        if (
            not hasattr(self, "_estimator_type")
            or self._estimator_type != "classifier"
        ):
            raise AttributeError(
                "predict_log_proba is not available for this estimator type."
            )

        check_is_fitted(self, ["rff_"])
        X = check_array(X)

        if not hasattr(self.estimator, "predict_log_proba"):
            raise ValueError(
                f"The estimator {type(self.estimator).__name__} does not support predict_log_proba."
            )

        # Transform the input data
        X_transformed = self.rff_.transform(X)

        return self.estimator.predict_log_proba(X_transformed)

    def decision_function(self, X):
        """
        Decision function (only for classifiers).
        """
        if (
            not hasattr(self, "_estimator_type")
            or self._estimator_type != "classifier"
        ):
            raise AttributeError(
                "decision_function is not available for this estimator type."
            )

        check_is_fitted(self, ["rff_"])
        X = check_array(X)

        if not hasattr(self.estimator, "decision_function"):
            raise ValueError(
                f"The estimator {type(self.estimator).__name__} does not support decision_function."
            )

        # Transform the input data
        X_transformed = self.rff_.transform(X)

        return self.estimator.decision_function(X_transformed)

    def score(self, X, y):
        """
        Evaluate the model performance.
        """
        check_is_fitted(self, ["rff_"])
        X = check_array(X)

        # Transform the input data
        X_transformed = self.rff_.transform(X)

        # Evaluate using the underlying estimator's score method
        return self.estimator.score(X_transformed, y)

    @property
    def classes_(self):
        """Classes labels (only for classifiers)."""
        if (
            hasattr(self, "_estimator_type")
            and self._estimator_type == "classifier"
        ):
            return getattr(self.estimator, "classes_", None)
        else:
            raise AttributeError(
                "classes_ is not available for this estimator type."
            )

    def get_params(self, deep=True):
        """
        Get parameters for this estimator.
        """
        params = {}

        # Get estimator parameters with proper prefixing
        if deep:
            estimator_params = self.estimator.get_params(deep=True)
            for key, value in estimator_params.items():
                params[f"estimator__{key}"] = value

        # Add our own parameters
        params.update(
            {
                "estimator": self.estimator,
                "n_components": self.n_components,
                "gamma": self.gamma,
                "random_state": self.random_state,
            }
        )

        return params

    def set_params(self, **params):
        """
        Set the parameters of this estimator.
        """
        # Separate our parameters from estimator parameters
        our_params = {}
        estimator_params = {}

        for param, value in params.items():
            if param.startswith("estimator__"):
                # Remove the 'estimator__' prefix
                estimator_params[param[11:]] = value
            elif param in [
                "estimator",
                "n_components",
                "gamma",
                "random_state",
            ]:
                our_params[param] = value
            else:
                # Assume it's an estimator parameter without prefix
                estimator_params[param] = value

        # Set our parameters
        for param, value in our_params.items():
            setattr(self, param, value)

        # If estimator changed, update the estimator type
        if "estimator" in our_params:
            self.__init__(
                self.estimator, self.n_components, self.gamma, self.random_state
            )

        # Set estimator parameters
        if estimator_params:
            self.estimator.set_params(**estimator_params)

        # If RFF parameters changed and model is fitted, we need to refit
        if hasattr(self, "rff_") and (
            "n_components" in our_params
            or "gamma" in our_params
            or "random_state" in our_params
        ):
            # Remove the fitted transformer so it gets recreated on next fit
            delattr(self, "rff_")

        return self
