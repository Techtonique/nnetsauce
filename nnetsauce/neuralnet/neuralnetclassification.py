import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from nnetsauce.multitask.simplemultitaskClassifier import SimpleMultitaskClassifier
from nnetsauce.neuralnet import NeuralNetRegressor


class NeuralNetClassifier(BaseEstimator, ClassifierMixin):
    """
    (Pretrained) Neural Network Classifier.

    Parameters
    ----------
    hidden_layer_sizes : tuple, default=(100,)
        The number of neurons in each hidden layer.
    max_iter : int, default=100
        The maximum number of iterations to train the model.
    learning_rate : float, default=0.01
        The learning rate for the optimizer.
    random_state : int, default=None
        The random state for the random number generator.
    """
    def __init__(self, hidden_layer_sizes=(100,), 
                 max_iter=100, learning_rate=0.01, 
                 weights=None, random_state=None):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.weights = weights
        self.random_state = random_state
        self.regr = None

    def fit(self, X, y):
        regressor = NeuralNetRegressor(hidden_layer_sizes=self.hidden_layer_sizes, 
                                       max_iter=self.max_iter, 
                                       learning_rate=self.learning_rate, 
                                       weights=self.weights, 
                                       random_state=self.random_state)
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
        return self.regr.predict_proba(X)
    
    def predict(self, X):
        return self.regr.predict(X)


