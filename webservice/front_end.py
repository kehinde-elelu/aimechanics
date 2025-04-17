import argparse
import requests
import sounddevice as sd
from scipy.io.wavfile import write
import tempfile
import time
import os
import serial
from datetime import datetime

# Define the server URL
url = "http://127.0.0.1:8001/"
color = {'normal': 'Green', 'early_fault':'Yellow', 'failure':'Red'}
audio_folder = "/Users/kehindeelelu/Documents/aimechanics/dataset/recorded_audio"

color = {
    'normal': 'green',
    'early_fault': 'yellow',  # optionally use another LED
    'failure': 'red'
}

# Replace with your Particle device info
PARTICLE_DEVICE_ID = 'e00fce687d5dac10819a7918'
PARTICLE_ACCESS_TOKEN = 'd22576aded5b7474f866a23fadf4ded7f969b2fd'
PARTICLE_FUNCTION = 'setColor'

# Record audio from the microphone
def record_audio(duration=5, sample_rate=44100):
    print("Recording...")
    audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
    sd.wait()  # Wait until recording is finished
    print("Recording complete.")
    return audio_data, sample_rate

# Save the recorded audio to a specific audio_folder
def save_audio_to_folder(audio_data, sample_rate, audio_folder, filename="recorded_audio.wav"):
    os.makedirs(audio_folder, exist_ok=True)
    file_path = os.path.join(audio_folder, filename)
    write(file_path, sample_rate, audio_data)
    print(f"Audio saved to {file_path}")
    return file_path

# # Send color to the bulb via serial communication
# def send_color_to_bulb(color):
#     try:
#         # Replace 'COM3' with the correct port for your microcontroller (e.g., '/dev/ttyUSB0' on Linux/Mac)
#         with serial.Serial('/dev/ttyUSB0', 9600, timeout=1) as ser:
#             ser.write((color + '\n').encode())  # Send color as a string
#             print(f"Sent color to bulb: {color}")
#     except Exception as e:
#         print(f"Error sending color to bulb: {e}")


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


# Main logic
if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Audio classification client.")
    parser.add_argument("--mode", choices=["file_path", "file_upload"], required=True, help="Mode of operation: 'file_path' or 'file_upload'")
    args = parser.parse_args()
    mode_extract = args.mode # 'file_path' || 'file_upload'

    ############# Main Logic #############
    try: 
        # Record audio for 5 seconds
        audio_data, sample_rate = record_audio(duration=5)
        print("Audio data shape:", audio_data.shape, sample_rate)

        if mode_extract == "file_path":
            # Save the audio to the specified audio_folder with a unique filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"audio_{timestamp}.wav"
            audio_file_path = save_audio_to_folder(audio_data, sample_rate, audio_folder, filename)

            response = requests.post(url+"predict_path", data={"file_path": audio_file_path})
                
            # Parse the JSON response
            response_data = response.json()
            print("Server response:", response_data)

            # Access the predicted class
            print("Predicted class:", response_data.get("prediction", {}).get("predicted_class"))
        
        if mode_extract == "file_upload":
            #upload as a temp file to the server
            with tempfile.NamedTemporaryFile(delete=True) as temp_file:
                write(temp_file.name, sample_rate, audio_data)
                temp_file.seek(0)

                # =========> Call the server API to classify the audio <========= #
                response = requests.post(url+"equip_diagnostic", files={"file": temp_file})
                response_data = response.json()
                print("Server response:", response_data)
                print("Predicted class:", response_data.get("prediction", {}).get("predicted_class"))  
                # =========> Return the API response <========= # 

                # =========> Extract the color based on the predicted class  <========= #
                print("Predicted class color:", color.get(response_data.get("prediction", {}).get("predicted_class")))

                # =========> Call the ECU/ESP32 for changing of bulb <========= #
                predicted_color = color.get(response_data.get("prediction", {}).get("predicted_class"))
                print(predicted_color)
                if predicted_color:
                    send_color_to_bulb(predicted_color)

    ############# End of Main Logic #############
    except Exception as e:
        print("An error occurred:", e)


# call the file 
# python webservice/front_end.py --mode file_path
# python webservice/front_end.py --mode file_upload