from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from pydub import AudioSegment
from joblib import load
import numpy as np
import io
import os
import tempfile

from classification import classify_audio

base_path = "/Users/kehindeelelu/Documents/aimechanics/dataset/"
data_dir = os.path.join(base_path, "equipment_sound_dataset")

app = FastAPI()
model = load(os.path.join(base_path, "models", "svm_model.joblib"))

@app.get("/")
async def root():
    return {"message": "Welcome to the Audio Classification API. Use the /predict endpoint to classify audio files."}

@app.post("/predict")
async def predict(file_path: str = Form(...)):
    try:
        # Ensure the file exists
        if not os.path.exists(file_path):
            return JSONResponse({"error": "File not found"}, status_code=400)
        
        # Call classify_audio with the provided file path
        clas_result = classify_audio(model, file_path, ['normal', 'early_fault', 'failure'])

        # Convert NumPy array to list for JSON serialization
        clas_result['probabilities'] = clas_result['probabilities'].tolist()

        # Return the prediction result
        return JSONResponse({"prediction": clas_result})
    except Exception as e:
        # Handle errors and return a 500 response
        return JSONResponse({"error": str(e)}, status_code=500)


