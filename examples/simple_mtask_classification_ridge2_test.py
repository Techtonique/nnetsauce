import os 
import nnetsauce as ns
import numpy as np
from sklearn.datasets import load_breast_cancer, make_regression
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split
from sklearn import metrics
from time import time

print("=== DIAGNOSTIC TEST FOR ElasticNet2Regressor ===\n")

# Test 1: Simple regression task
print("1. Testing on simple regression task...")
X, y = make_regression(n_samples=200, n_features=10, noise=0.1, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Test with very weak regularization first
regr = ns.ElasticNet2Regressor(
    lambda1=0.001,  # Very weak regularization
    lambda2=0.001, 
    l1_ratio1=0.5,
    l1_ratio2=0.5,
    max_iter=1000,
    tol=1e-6,
    verbose=True
)

print("Fitting ElasticNet2Regressor...")
try:
    start = time()
    regr.fit(X_train, y_train)
    print(f"Fit completed in {time() - start:.2f} seconds")
    
    # Check if coefficients are all zero
    print(f"Number of non-zero coefficients: {np.sum(np.abs(regr.beta_) > 1e-10)}")
    print(f"Coefficient range: [{np.min(regr.beta_):.6f}, {np.max(regr.beta_):.6f}]")
    
    # Make predictions
    y_pred = regr.predict(X_test)
    print(f"Prediction range: [{np.min(y_pred):.6f}, {np.max(y_pred):.6f}]")
    print(f"True target range: [{np.min(y_test):.6f}, {np.max(y_test):.6f}]")
    
    # R² score
    r2 = regr.score(X_test, y_test)
    print(f"R² score: {r2:.4f}")
    
except Exception as e:
    print(f"ERROR: {e}")
    print("ElasticNet2Regressor failed - there's definitely a bug in the implementation")

print("\n" + "="*50)

# Test 2: Compare with sklearn ElasticNet
print("2. Comparing with sklearn ElasticNet...")
sklearn_regr = ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=1000)
sklearn_regr.fit(X_train, y_train)
sklearn_pred = sklearn_regr.predict(X_test)
sklearn_r2 = sklearn_regr.score(X_test, y_test)

print(f"Sklearn ElasticNet R² score: {sklearn_r2:.4f}")
print(f"Sklearn coefficients range: [{np.min(sklearn_regr.coef_):.6f}, {np.max(sklearn_regr.coef_):.6f}]")

print("\n" + "="*50)

# Test 3: Test the classification wrapper with a working regressor
print("3. Testing SimpleMultitaskClassifier with sklearn regressor...")
breast_cancer = load_breast_cancer()
Z = breast_cancer.data
t = breast_cancer.target

X_train, X_test, y_train, y_test = train_test_split(Z, t, test_size=0.2, random_state=42)

# Use sklearn ElasticNet as base regressor
sklearn_regr = ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=1000)
fit_obj = ns.SimpleMultitaskClassifier(sklearn_regr)

start = time()
fit_obj.fit(X_train, y_train)
print(f"Sklearn-based classifier fit time: {time() - start:.2f} seconds")

preds = fit_obj.predict(X_test)
unique_preds = np.unique(preds)
print(f"Unique predictions: {unique_preds}")
print(f"Accuracy: {metrics.accuracy_score(y_test, preds):.4f}")

if len(unique_preds) > 1:
    print("SUCCESS: Sklearn-based classifier predicts multiple classes")
    print(metrics.classification_report(y_test, preds))
else:
    print("WARNING: Even sklearn-based classifier only predicts one class")

print("\n" + "="*50)

# Test 4: Try ElasticNet2Regressor with different parameters
print("4. Testing ElasticNet2Regressor with different parameters...")

param_sets = [
    {"lambda1": 0.0, "lambda2": 0.0, "description": "No regularization"},
    {"lambda1": 0.1, "lambda2": 0.0, "l1_ratio1": 0.0, "l1_ratio2": 0.0, "description": "Pure L2"},
    {"lambda1": 0.01, "lambda2": 0.01, "solver": "adam", "description": "Adam optimizer"},
]

for i, params in enumerate(param_sets, 1):
    description = params.pop("description")
    print(f"\n4.{i} {description}:")
    
    regr = ns.ElasticNet2Regressor(**params, verbose=False, max_iter=500)
    
    try:
        regr.fit(X_train, y_train)
        y_pred = regr.predict(X_test)
        r2 = regr.score(X_test, y_test)
        
        # Test in classification wrapper
        fit_obj = ns.SimpleMultitaskClassifier(regr)
        fit_obj.fit(X_train, y_train)
        preds = fit_obj.predict(X_test)
        unique_preds = np.unique(preds)
        
        print(f"  Regression R²: {r2:.4f}")
        print(f"  Classification unique predictions: {unique_preds}")
        print(f"  Classification accuracy: {metrics.accuracy_score(y_test, preds):.4f}")
        
    except Exception as e:
        print(f"  ERROR: {e}")

print("\n=== DIAGNOSTIC COMPLETE ===")