import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from nnetsauce.multitask.simplemultitaskClassifier import (
    SimpleMultitaskClassifier,
)
from nnetsauce.neuralnet import NeuralNetRegressor


class NeuralNetClassifier(BaseEstimator, ClassifierMixin):
    """
    (Pretrained) Neural Network Classifier.

    Parameters:

        hidden_layer_sizes : tuple, default=(100,)
            The number of neurons in each hidden layer.
        max_iter : int, default=100
            The maximum number of iterations to train the model.
        learning_rate : float, default=0.01
            The learning rate for the optimizer.
        l1_ratio : float, default=0.5
            The ratio of L1 regularization.
        alpha : float, default=1e-6
            The regularization parameter.
        activation_name : str, default="relu"
            The activation function to use.
        dropout : float, default=0.0
            The dropout rate.
        random_state : int, default=None
            The random state for the random number generator.
        weights : list, default=None
            The weights to initialize the model with.

    Attributes:

        weights : list
            The weights of the model.
        params : list
            The parameters of the model.
        scaler_ : sklearn.preprocessing.StandardScaler
            The scaler used to standardize the input features.
        y_mean_ : float
            The mean of the target variable.

    Methods:

        fit(X, y)
            Fit the model to the data.
        predict(X)
            Predict the target variable.
        predict_proba(X)
            Predict the probability of the target variable.
        get_weights()
            Get the weights of the model.
        set_weights(weights)
            Set the weights of the model.
    """

    def __init__(
        self,
        hidden_layer_sizes=(100,),
        max_iter=100,
        learning_rate=0.01,
        weights=None,
        l1_ratio=0.5,
        alpha=1e-6,
        activation_name="relu",
        dropout=0.0,
        random_state=None,
    ):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.weights = weights
        self.l1_ratio = l1_ratio
        self.alpha = alpha
        self.activation_name = activation_name
        self.dropout = dropout
        self.random_state = random_state
        self.regr = None

    def fit(self, X, y):
        """Fit the model to the data.

        Parameters:

            X: {array-like}, shape = [n_samples, n_features]
                Training vectors, where n_samples is the number of samples and
                n_features is the number of features.
            y: array-like, shape = [n_samples]
                Target values.
        """
        regressor = NeuralNetRegressor(
            hidden_layer_sizes=self.hidden_layer_sizes,
            max_iter=self.max_iter,
            learning_rate=self.learning_rate,
            weights=self.weights,
            l1_ratio=self.l1_ratio,
            alpha=self.alpha,
            activation_name=self.activation_name,
            dropout=self.dropout,
            random_state=self.random_state,
        )
        self.regr = SimpleMultitaskClassifier(regressor)
        self.regr.fit(X, y)
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        self.n_tasks_ = 1
        self.n_features_in_ = X.shape[1]
        self.n_outputs_ = 1
        self.n_samples_fit_ = X.shape[0]
        self.n_samples_test_ = X.shape[0]
        self.n_features_out_ = 1
        self.n_outputs_ = 1
        self.n_features_in_ = X.shape[1]
        self.n_features_out_ = 1
        self.n_outputs_ = 1
        return self

    def predict_proba(self, X):
        """Predict the probability of the target variable.

        Parameters:

            X: {array-like}, shape = [n_samples, n_features]
                Training vectors, where n_samples is the number of samples and
                n_features is the number of features.
        """
        return self.regr.predict_proba(X)

    def predict(self, X):
        """Predict the target variable.

        Parameters:

            X: {array-like}, shape = [n_samples, n_features]
                Training vectors, where n_samples is the number of samples and
                n_features is the number of features.
        """
        return self.regr.predict(X)
