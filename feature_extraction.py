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


# 2. Feature Extraction
def extract_features(audio_data):
    """Extract audio features from a list of audio files"""
    features = []
    
    for audio, sr in audio_data:
        # Apply preprocessing (normalize audio)
        audio = librosa.util.normalize(audio)
        
        # Set frame length and hop length
        frame_length = int(sr * 0.025)  # 25ms frame
        hop_length = int(sr * 0.01)     # 10ms hop
        
        # Extract features
        feature_vector = []
        
        # Time domain features
        feature_vector.append(np.mean(np.abs(audio)))  # Mean absolute amplitude
        feature_vector.append(np.std(audio))           # Standard deviation
        feature_vector.append(np.max(np.abs(audio)))   # Peak amplitude
        feature_vector.append(np.sum(np.abs(np.diff(np.sign(audio)) > 0)) / len(audio))  # Zero crossing rate
        
        # Spectral features
        # MFCCs
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        feature_vector.extend(np.mean(mfccs, axis=1))
        feature_vector.extend(np.std(mfccs, axis=1))
        
        # Spectral centroid
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
        feature_vector.append(np.mean(spectral_centroid))
        feature_vector.append(np.std(spectral_centroid))
        
        # Spectral bandwidth
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)[0]
        feature_vector.append(np.mean(spectral_bandwidth))
        feature_vector.append(np.std(spectral_bandwidth))
        
        # Spectral rolloff
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]
        feature_vector.append(np.mean(spectral_rolloff))
        feature_vector.append(np.std(spectral_rolloff))
        
        # Chroma features
        chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
        feature_vector.extend(np.mean(chroma, axis=1))
        
        # Additional: Spectral contrast
        contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
        feature_vector.extend(np.mean(contrast, axis=1))
        
        features.append(feature_vector)
    
    return np.array(features)