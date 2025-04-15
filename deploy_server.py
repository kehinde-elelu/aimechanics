from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from pydub import AudioSegment
from joblib import load
import numpy as np
import io

from classification import classify_audio

app = FastAPI()
model = load("svm_model.joblib")

def extract_features(audio_bytes):
    audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format="m4a")
    samples = np.array(audio.get_array_of_samples()).astype(np.float32)
    # Add actual feature extraction logic here
    features = np.mean(samples), np.std(samples)
    return np.array(features).reshape(1, -1)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    audio_bytes = await file.read()
    features = extract_features(audio_bytes)
    prediction = model.predict(features)[0]
    return JSONResponse({"prediction": prediction})

classify_audio(model, f"{data_dir}/normal/normal_001.wav", ['normal', 'early_fault', 'failure'])
