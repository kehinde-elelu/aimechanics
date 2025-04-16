import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import librosa
import librosa.display
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
import os
import warnings
warnings.filterwarnings('ignore')


# 4. Build and train the SVM model
def build_svm_model(X_train, y_train):
    """Build and optimize an SVM classifier"""
    # Create a pipeline with scaling, dimensionality reduction, and SVM
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=0.95)),  # Keep 95% of variance
        ('svm', SVC(probability=True))
    ])
    
    # Parameters for grid search
    param_grid = {
        'svm__C': [0.1, 1, 10, 100],
        'svm__gamma': ['scale', 'auto', 0.01, 0.1, 1],
        'svm__kernel': ['rbf', 'poly']
    }
    
    # Grid search with cross-validation
    grid = GridSearchCV(
        pipeline, param_grid, cv=5, scoring='f1_weighted', verbose=1, n_jobs=-1
    )
    
    # Train the model
    print("Training SVM model with grid search...")
    grid.fit(X_train, y_train)
    
    print(f"Best parameters: {grid.best_params_}")
    print(f"Best cross-validation score: {grid.best_score_:.4f}")
    
    return grid.best_estimator_