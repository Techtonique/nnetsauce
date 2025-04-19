import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from nnetsauce.kernel.kernel import KernelRidge
import numpy as np

# Load the Diabetes dataset
dataset = load_diabetes()
X, y = dataset.data, dataset.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the KernelRidge model with Nyström approximation
model = KernelRidge(
    alpha=10**np.linspace(-3, 3, 10), 
    kernel="matern", 
    gamma=0.1, 
    nu=1.5, 
    use_nystrom=True, 
    n_landmarks=100
)

# Fit the model
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print(f"RMSE with Nyström Approximation: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")

# Load the Boston housing dataset
df = pd.read_csv(
    "https://raw.githubusercontent.com/selva86/datasets/refs/heads/master/BostonHousing.csv"
)
X = df.drop(columns=["medv"]).values
y = df["medv"].values

# Split the data into training and testing sets
X_train_, X_test_, y_train_, y_test_ = train_test_split(X, y, test_size=0.8, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X_test_, y_test_, test_size=0.2, random_state=42)

# Initialize the KernelRidge model with Nyström approximation
model = KernelRidge(
    alpha=10**np.linspace(-3, 3, 10), 
    kernel="matern", 
    gamma=0.1, 
    nu=1.5, 
    use_nystrom=True, 
    n_landmarks=100
).fit(X_train_, y_train_)

# Evaluate the model on the test set
y_pred = model.predict(X_test)
print(f"RMSE with Nyström Approximation: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")

# Incremental training using partial_fit
batch_size = 50
num_batches = X_train.shape[0] // batch_size

for i in range(num_batches):
    # Get the current batch
    start_idx = i * batch_size
    end_idx = start_idx + batch_size
    X_batch = X_train[start_idx:end_idx]
    y_batch = y_train[start_idx:end_idx]

    # Use partial_fit to update the model
    model.partial_fit(X_batch, y_batch)

    # Evaluate the model after each batch
    y_pred = model.predict(X_test)
    mse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"Batch {i + 1}/{num_batches} - Updated RMSE with Nyström Approximation: {mse:.2f}")

# Final evaluation after all batches
y_pred_final = model.predict(X_test)
final_mse = np.sqrt(mean_squared_error(y_test, y_pred_final))
print(f"Final RMSE with Nyström Approximation after all batches: {final_mse:.2f}")
