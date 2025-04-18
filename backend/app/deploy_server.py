'''
# run the server using uvicorn
uvicorn backend.deploy_server:app --reload
'''

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from pydub import AudioSegment
from joblib import load
import numpy as np
import io
import os
import tempfile
from scipy.io.wavfile import write
import requests
from svm_.classification import classify_audio

base_path = "/app/"
data_dir = os.path.join(base_path, "equipment_sound_dataset")
sample_rate = 44100  # Replace with the actual sample rate of your audio


##############################################
color = {
    'normal': 'green',
    'early_fault': 'yellow',  # optionally use another LED
    'failure': 'red'
}

# Replace with your Particle device info
PARTICLE_DEVICE_ID = 'e00fce687d5dac10819a7918'
PARTICLE_ACCESS_TOKEN = 'd22576aded5b7474f866a23fadf4ded7f969b2fd'
PARTICLE_FUNCTION = 'setColor'

def send_color_to_bulb(color):
    try:
        url = f"https://api.particle.io/v1/devices/{PARTICLE_DEVICE_ID}/{PARTICLE_FUNCTION}"
        print(url)
        data = {
            'arg': color,
            'access_token': PARTICLE_ACCESS_TOKEN
        }
        response = requests.post(url, data=data)
        print(response.status_code)
        print(response.text)
        if response.status_code == 200:
            print(f"Bulb color set to: {color}")
        else:
            print("Failed to send color to bulb:", response.text)
    except Exception as e:
        print("Error sending color to bulb:", e)
##############################################


app = FastAPI()
model = load(os.path.join(base_path, "models", "svm_model.joblib"))

@app.get("/")
async def root():
    return {"message": "Welcome to the Audio Classification API. Use the /predict endpoint to classify audio files."}

@app.post("/predict_path")
async def predict(file_path: str = Form(...)):
    print("==========================================")
    print("URL for uploading audio data to the local path")
    print("==========================================")
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
    

@app.post("/equip_diagnostic")
async def predict(file: UploadFile = File(...)):
    print("==========================================")
    print("URL for uploading audio data to the server")
    print("==========================================")
    try:
        # Save the uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            temp_file_path = temp_file.name
            temp_file.write(await file.read())

        # =========> Call the SVM Model for classification <========= #
        # Call classify_audio with the temporary file path
        class_result = classify_audio(model, temp_file_path, ['normal', 'early_fault', 'failure']) 
        # =========> Return the classification <================ # 

        # Convert NumPy array to list for JSON serialization
        class_result['probabilities'] = class_result['probabilities'].tolist()

        # write the temporary file to the f'{base_path}/dataset/recorded_audio/' directory
        audio_data = AudioSegment.from_file(temp_file_path).get_array_of_samples()
        os.makedirs(os.path.join(base_path, 'recorded_audio'), exist_ok=True)
        output_file_path = os.path.join(base_path, 'recorded_audio', 'output_audio.wav')
        write(output_file_path, sample_rate, np.array(audio_data))

        # Clean up the temporary file
        os.remove(temp_file_path)

        predicted_class = class_result['predicted_class']
        predicted_color = {
            'normal': 'green',
            'early_fault': 'yellow',
            'failure': 'red'
        }.get(predicted_class)
        send_color_to_bulb(predicted_color)

        # Return the prediction result
        return JSONResponse({"prediction": class_result})
    except Exception as e:
        # Handle errors and return a 500 response
        return JSONResponse({"error": str(e)}, status_code=500)


'''
    Docker Deployment
    docker build -t backend .
    docker run -d -p 8000:8000 backend
'''
