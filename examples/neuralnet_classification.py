import numpy as np
from nnetsauce.neuralnet import NeuralNetClassifier
from sklearn.datasets import load_breast_cancer, load_iris, load_wine, load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from time import time


datasets = [load_breast_cancer(), load_iris(), load_wine(), load_digits()]

dataset_names = ['Breast Cancer', 'Iris', 'Wine', 'Digits']

for dataset, name in zip(datasets, dataset_names):

    print(f"\n\n Dataset: {name} ------------------------------")

    X, y = dataset.data, dataset.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.1, 
                                                    random_state=42)

    X_train_1, X_train_2, y_train_1, y_train_2 = train_test_split(X_train, y_train, 
                                                              test_size=0.5, 
                                                              random_state=42)
    # train with random initial weights
    model = NeuralNetClassifier(hidden_layer_sizes=(100,), 
                                max_iter=100, learning_rate=0.01, 
                                random_state=42)
    start_time = time()
    model.fit(X_train_1, y_train_1)
    end_time = time()
    print(f"Time taken to fit: {end_time - start_time} seconds")
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

    # train with pretrained weights
    model.set_params(weights=model.weights)
    start_time = time()
    model.fit(X_train_2, y_train_2)
    end_time = time()
    print(f"Time taken to fit: {end_time - start_time} seconds")
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

