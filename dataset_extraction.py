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


# 1. Data Loading Function
def load_audio_files(data_dir):
    """
    Load audio files from directory structure:
    data_dir/
        normal/
            file1.wav
            file2.wav
            ...
        early_fault/
            file1.wav
            ...
        failure/
            file1.wav
            ...
    """
    X = []  # Features will go here
    y = []  # Labels will go here
    classes = {'normal': 0, 'early_fault': 1, 'failure': 2}
    
    for condition in classes.keys():
        path = os.path.join(data_dir, condition)
        
        # Skip if directory doesn't exist
        if not os.path.isdir(path):
            continue
            
        for file in os.listdir(path):
            if file.endswith('.wav'):
                file_path = os.path.join(path, file)
                # Load audio file
                try:
                    audio, sr = librosa.load(file_path, sr=None)
                    X.append((audio, sr))
                    y.append(classes[condition])
                    print(f"Loaded {file_path}")
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
    
    return X, y