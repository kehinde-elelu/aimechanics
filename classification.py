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

from feature_extraction import extract_features

# 7. Function to classify new audio
def classify_audio(model, audio_file, class_names):
    """Classify a single audio file using the trained model"""
    try:
        # Load audio
        audio, sr = librosa.load(audio_file, sr=None)
        
        # Extract features (same as in training)
        audio_data = [(audio, sr)]
        features = extract_features(audio_data)
        print(features.shape)
        
        # Make prediction
        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]

        result = {
            'predicted_class': class_names[prediction],
            'probabilities': probabilities
        }
        
        # Print results
        print(f"\nClassification for {os.path.basename(audio_file)}:")
        print(f"Predicted class: {class_names[prediction]}")
        print("Class probabilities:")
        for i, class_name in enumerate(class_names):
            print(f"  {class_name}: {probabilities[i]:.4f}")
        
    except Exception as e:
        print(f"Error classifying audio: {e}")
    return result