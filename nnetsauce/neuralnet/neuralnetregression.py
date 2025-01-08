try:
    import jax 
    import jax.numpy as jnp
    from jax import grad, jit, vmap
except ImportError:
    raise ImportError("Jax is not installed. Please install Jax to use this module.")

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler

class NeuralNetRegressor(BaseEstimator, RegressorMixin):
    """
    (Pretrained) Neural Network Regressor.

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
    weights : list, default=None
        The weights to initialize the model with.

    Attributes
    ----------
    weights : list
        The weights of the model.
    params : list
        The parameters of the model.
    scaler_ : sklearn.preprocessing.StandardScaler
        The scaler used to standardize the input features.
    y_mean_ : float
        The mean of the target variable.

    Methods
    -------
    fit(X, y)
        Fit the model to the data.
    predict(X)
        Predict the target variable.
    get_weights()
        Get the weights of the model.
    set_weights(weights)
        Set the weights of the model.

    """
    def __init__(self, hidden_layer_sizes=None, 
                 max_iter=100, learning_rate=0.01,
                 weights=None, random_state=None):
        if weights is None and hidden_layer_sizes is None:
            hidden_layer_sizes = (100,)  # default value if neither is provided
        self.hidden_layer_sizes = hidden_layer_sizes
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.weights = weights
        self.random_state = random_state        
        self.params = None
        self.scaler_ = StandardScaler()
        self.y_mean_ = None

    def _validate_weights(self, input_dim):
        """Validate that weights dimensions are coherent."""
        if not self.weights:
            return False
            
        try:
            # Check each layer's weights and biases
            prev_dim = input_dim
            for W, b in self.weights:
                # Check weight matrix dimensions
                if W.shape[0] != prev_dim:
                    raise ValueError(
                        f"Weight matrix input dimension {W.shape[0]} does not match "
                        f"previous layer output dimension {prev_dim}"
                    )
                # Check bias dimension matches weight matrix output
                if W.shape[1] != b.shape[0]:
                    raise ValueError(
                        f"Bias dimension {b.shape[0]} does not match weight matrix "
                        f"output dimension {W.shape[1]}"
                    )
                prev_dim = W.shape[1]
            
            # Check final output dimension is 1 for regression
            if prev_dim != 1:
                raise ValueError(
                    f"Final layer output dimension {prev_dim} must be 1 for regression"
                )
            
            return True
        except (AttributeError, IndexError):
            raise ValueError("Weights format is invalid. Expected list of (weight, bias) tuples")

    def fit(self, X, y):
        # Standardize the input features
        X = self.scaler_.fit_transform(X)
        # Ensure y is 2D for consistency
        y = y.reshape(-1, 1)
        self.y_mean_ = jnp.mean(y)
        y = y - self.y_mean_        
        # Validate or initialize weights
        if self.weights is not None:
            if self._validate_weights(X.shape[1]):
                self.params = self.weights
        else:
            if self.hidden_layer_sizes is None:
                raise ValueError("Either weights or hidden_layer_sizes must be provided")
            self.params = initialize_params(X.shape[1], 
                                            self.hidden_layer_sizes, 
                                            self.random_state)        
        # Training loop
        for _ in range(self.max_iter):
            grads = perex_grads(self.params, X, y)
            # Average gradients across examples
            grads = jax.tree_map(lambda g: jnp.mean(g, axis=0), grads)
            # Update parameters
            self.params = [(W - self.learning_rate * dW, b - self.learning_rate * db)
                          for (W, b), (dW, db) in zip(self.params, grads)]            
        # Store final weights
        self.weights = self.params
        return self

    def get_weights(self):
        """Return the current weights of the model."""
        if self.weights is None:
            raise ValueError("No weights available. Model has not been fitted yet.")
        return self.weights

    def set_weights(self, weights):
        """Set the weights of the model manually."""
        self.weights = weights
        self.params = weights

    def predict(self, X):
        X = self.scaler_.transform(X)        
        if self.params is None:
            raise ValueError("Model has not been fitted yet.")
        predictions = predict(self.params, X)
        return predictions.reshape(-1) + self.y_mean_


def predict(params, inputs):
  for W, b in params:
    outputs = jnp.dot(inputs, W) + b
    inputs = jnp.tanh(outputs)  # inputs to the next layer
  return outputs                # no activation on last layer

def loss(params, inputs, targets):
  preds = predict(params, inputs)
  return jnp.sum((preds - targets)**2)

grad_loss = jit(grad(loss))  # compiled gradient evaluation function
perex_grads = jit(vmap(grad_loss, in_axes=(None, 0, 0)))  # fast per-example grads

def initialize_params(input_dim, hidden_layer_sizes, random_state=None):
    """Initialize network parameters."""
    if random_state is not None:
        rng = jax.random.PRNGKey(random_state)
    else:
        rng = jax.random.PRNGKey(0)
    
    # Layer dimensions including input and output
    layer_sizes = [input_dim] + list(hidden_layer_sizes) + [1]
    
    # Initialize parameters for each layer
    params = []
    for i in range(len(layer_sizes) - 1):
        rng, key = jax.random.split(rng)
        # Xavier/Glorot initialization
        scale = jnp.sqrt(2.0 / (layer_sizes[i] + layer_sizes[i+1]))
        W = scale * jax.random.normal(key, (layer_sizes[i], layer_sizes[i+1]))
        b = jnp.zeros(layer_sizes[i+1])
        params.append((W, b))
    return params
