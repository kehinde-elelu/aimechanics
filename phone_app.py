import requests
import sounddevice as sd
from scipy.io.wavfile import write
import tempfile
import time
import os
from datetime import datetime

# Define the server URL
url = "http://127.0.0.1:8000/predict"

# Define the folder to save audio files
audio_folder = "/Users/kehindeelelu/Documents/aimechanics/dataset/recorded_audio"

# Record audio from the microphone
def record_audio(duration=5, sample_rate=44100):
    print("Recording...")
    audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
    sd.wait()  # Wait until recording is finished
    print("Recording complete.")
    return audio_data, sample_rate

# Save the recorded audio to a specific folder
def save_audio_to_folder(audio_data, sample_rate, folder, filename="recorded_audio.wav"):
    os.makedirs(folder, exist_ok=True)
    file_path = os.path.join(folder, filename)
    write(file_path, sample_rate, audio_data)
    print(f"Audio saved to {file_path}")
    return file_path

# Main logic
if __name__ == "__main__":
    try: 
        # Record audio for 5 seconds
        audio_data, sample_rate = record_audio(duration=5)
        print("Audio data shape:", audio_data.shape)

        # Save the audio to the specified folder with a unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"audio_{timestamp}.wav"
        audio_file_path = save_audio_to_folder(audio_data, sample_rate, audio_folder, filename)

        response = requests.post(url, data={"file_path": audio_file_path})
            
        # Parse the JSON response
        response_data = response.json()
        print("Server response:", response_data)

        # Access the predicted class
        print("Predicted class:", response_data.get("prediction", {}).get("predicted_class"))
    except Exception as e:
        print("An error occurred:", e)