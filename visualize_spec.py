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


# 3. Visualize features
def visualize_audio(audio_data, labels, class_names, n_samples=3):
    """Visualize waveforms and spectrograms for a few samples from each class"""
    fig, axes = plt.subplots(nrows=len(class_names), ncols=2*n_samples, figsize=(20, 10))
    
    for i, class_label in enumerate(range(len(class_names))):
        # Get samples for this class
        indices = [j for j, label in enumerate(labels) if label == class_label]
        if len(indices) >= n_samples:
            samples = np.random.choice(indices, n_samples, replace=False)
            
            for j, sample_idx in enumerate(samples):
                audio, sr = audio_data[sample_idx]
                
                # Plot waveform
                col = 2*j
                axes[i, col].set_title(f'{class_names[class_label]} - Sample {j+1} Waveform')
                librosa.display.waveshow(audio, sr=sr, ax=axes[i, col])
                
                # Plot spectrogram
                col = 2*j + 1
                axes[i, col].set_title(f'{class_names[class_label]} - Sample {j+1} Spectrogram')
                D = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
                librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log', ax=axes[i, col])
    
    plt.tight_layout()
    plt.show()