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
from sklearn import svm
from joblib import dump, load
import warnings
warnings.filterwarnings('ignore')

from dataset_extraction import load_audio_files
from visualize_spec import visualize_audio
from feature_extraction import extract_features
from svm_model import build_svm_model
from evaluation import evaluate_model
from classification import classify_audio

base_path = "/Users/kehindeelelu/Documents/aimechanics/dataset/"

# 6. Main function to run the entire pipeline
def main(data_dir, visualize=True):
    # Define class names
    class_names = ['normal', 'early_fault', 'failure']
    
    # 1. Load data
    print("Loading audio files...")
    audio_data, labels = load_audio_files(data_dir)
    
    if len(audio_data) == 0:
        print("No audio files found. Please check the directory path.")
        return
    
    # # 2. Visualize a few samples (optional)
    # if visualize:
    #     visualize_audio(audio_data, labels, class_names)
    
    # 3. Extract features
    print("Extracting features...")
    features = extract_features(audio_data)
    
    # 4. Split data
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.25, random_state=42, stratify=labels
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Testing set: {X_test.shape[0]} samples")
    
    # 5. Build and train the model
    model = build_svm_model(X_train, y_train)
    
    # 6. Evaluate the model
    accuracy = evaluate_model(model, X_test, y_test, class_names)
    print(f"\nOverall accuracy: {accuracy:.4f}")

    # Save the model
    save_folder = os.path.join(base_path, "models")
    os.makedirs(save_folder, exist_ok=True)
    if model:
        dump(model, os.path.join(save_folder, "svm_model.joblib"))
    
    return model


# Usage example
if __name__ == "__main__":
    # Replace with your data directory
    model_train = True
    model_classify = True 
    data_dir = os.path.join(base_path, "equipment_sound_dataset")
    
    print("\n\n ==========Model Training and Evaluation==========")
    if model_train:
        # Train the model
        model = main(data_dir)
    
    print("\n\n ==========Model classification==========")
    if model_classify:
        model = load(os.path.join(base_path, "models", "svm_model.joblib"))
        classify_audio(model, f"{data_dir}/normal/normal_001.wav", ['normal', 'early_fault', 'failure'])