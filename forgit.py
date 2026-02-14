"""
Project: MNIST Digit Classification Pipeline
Author: apswa0407
Description: Implementing an optimized Support Vector Machine (SVM) 
             to recognize handwritten digits from the MNIST dataset.
"""

import sys
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

def load_data():
    """
    Fetch and load the MNIST dataset.
    Using a subset of 10,000 samples for optimal performance.
    """
    print("[INFO] Fetching MNIST dataset from OpenML...")
    try:
        # Load data using auto-parser for compatibility
        mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
        X, y = mnist["data"][:10000], mnist["target"][:10000]
        return X, y
    except Exception as e:
        print(f"[ERROR] Failed to load dataset: {e}")
        sys.exit(1)

def preprocess_pipeline(X, y):
    """
    Split the data and apply feature scaling.
    Standardization is crucial for SVM performance.
    """
    print("[INFO] Splitting data and applying StandardScaler...")
    
    # 80% Training, 20% Testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scaling pixel values (0-255) to a standard normal distribution
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test

def train_classifier(X_train, y_train):
    """
    Initialize and train the Support Vector Classifier.
    Using a Polynomial kernel for better non-linear boundary detection.
    """
    print("[INFO] Training the SVM Classifier (Kernel='poly')...")
    clf = SVC(kernel='poly', degree=3, gamma='scale')
    clf.fit(X_train, y_train)
    return clf

def run_evaluation(model, X_test, y_test):
    """
    Evaluate the model using Accuracy, Classification Report, and Confusion Matrix.
    """
    print("[INFO] Evaluating model performance...")
    predictions = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, predictions)
    
    print("\n" + "="*50)
    print(f"REPORT: MNIST DIGIT CLASSIFICATION")
    print("="*50)
    print(f"Overall Accuracy: {accuracy * 100:.2f}%")
    print("\nFull Metrics:\n")
    print(classification_report(y_test, predictions))
    print("="*50)

if __name__ == "__main__":
    # Main execution flow
    X_raw, y_raw = load_data()
    X_tr, X_te, y_tr, y_te = preprocess_pipeline(X_raw, y_raw)
    
    svm_model = train_classifier(X_tr, y_tr)
    run_evaluation(svm_model, X_te, y_te)
    
    print("\n[SUCCESS] Pipeline executed without errors.")
    