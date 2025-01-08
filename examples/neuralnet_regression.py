import numpy as np
from nnetsauce.neuralnet import NeuralNetRegressor
from sklearn.datasets import load_diabetes, fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from time import time

X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.2, 
                                                    random_state=42)

X_train_1, X_train_2, y_train_1, y_train_2 = train_test_split(X_train, y_train, 
                                                              test_size=0.5, 
                                                              random_state=42)

model = NeuralNetRegressor(hidden_layer_sizes=(100,), 
                           max_iter=100, 
                           learning_rate=0.01, 
                           random_state=42)
start_time = time()
model.fit(X_train_1, y_train_1)
end_time = time()
print(f"Time taken to fit model: {end_time - start_time:.2f} seconds")

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Root Mean Squared Error: {np.sqrt(mse):.4f}")

model2 = NeuralNetRegressor(max_iter=100, 
                           learning_rate=0.01, 
                           random_state=42,
                           weights=model.get_weights())
start_time = time()
model2.fit(X_train_2, y_train_2)
end_time = time()
print(f"Time taken to fit model: {end_time - start_time:.2f} seconds")

y_pred = model2.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Root Mean Squared Error: {np.sqrt(mse):.4f}")
