import os 
import nnetsauce as ns 
import numpy as np
import sklearn.metrics as skm2
from sklearn.datasets import load_breast_cancer, load_iris, load_wine, load_digits
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, RidgeClassifierCV, SGDClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from time import time 
from sklearn.metrics import classification_report
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt

print(f"\n ----- Running: {os.path.basename(__file__)}... ----- \n")

def plot_calibration_curve(y_true, y_prob, n_bins=10, title='Calibration Plot'):
    """Plot calibration curve for a single class."""
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)
    
    plt.figure(figsize=(8, 6))
    plt.plot(prob_pred, prob_true, 's-', label='Calibration curve')
    plt.plot([0, 1], [0, 1], '--', label='Perfectly calibrated')
    plt.xlabel('Mean predicted probability')
    plt.ylabel('True probability')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

load_models = [load_wine, load_breast_cancer, load_iris, load_digits]
dataset_names = ["wine", "breast_cancer", "iris", "digits"]

print("Example 1 - without weights")

for i, model in enumerate(load_models): 

    data = model()
    X = data.data
    y= data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .3, random_state = 13)

    obj = LogisticRegression()

    clf = ns.DeepClassifier(obj, n_layers=2, verbose=1, n_clusters=2, n_hidden_features=2)

    start = time()
    clf.fit(X_train, y_train)
    print(f"\nElapsed: {time() - start} seconds\n")

    preds = clf.predict(X_test)
    print("\n ----- dataset: ", dataset_names[i])
    print(classification_report(y_test, y_pred=preds))

    # Plot calibration curves for each class
    probs = clf.predict_proba(X_test)
    for class_idx in range(probs.shape[1]):
        class_name = f"Class {class_idx}"
        y_true_class = (y_test == class_idx).astype(int)
        plot_calibration_curve(y_true_class, probs[:, class_idx], 
                             title=f'Calibration Plot - {dataset_names[i]} - simple calib - {class_name}')

print("Example 2 - with weights")


for i, model in enumerate(load_models): 

    data = model()
    X = data.data
    y= data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .3, random_state = 13)

    obj = LogisticRegression()

    clf = ns.DeepClassifier(obj, n_layers=2, verbose=1, n_clusters=2, n_hidden_features=2)

    start = time()
    clf.fit(X_train, y_train, sample_weight=np.random.rand(X_train.shape[0]))
    print(f"\nElapsed: {time() - start} seconds\n")

    preds = clf.predict(X_test)
    print("\n ----- dataset: ", dataset_names[i])
    print(classification_report(y_test, y_pred=preds))

    # Plot calibration curves for each class
    probs = clf.predict_proba(X_test)
    for class_idx in range(probs.shape[1]):
        class_name = f"Class {class_idx}"
        y_true_class = (y_test == class_idx).astype(int)
        plot_calibration_curve(y_true_class, probs[:, class_idx], 
                             title=f'Calibration Plot - {dataset_names[i]} - simple calib - {class_name}')