# Aimechanics Backend

## Overview
The Aimechanics Backend project is a FastAPI application designed for audio classification. It provides endpoints for predicting audio classifications based on file paths and uploaded audio files. The application utilizes a machine learning model to classify audio data into different categories.

## Project Structure
```
backend
├── app
│   └── deploy_server.py      # FastAPI application for audio classification
├── Dockerfile                 # Dockerfile for building the application image
├── requirements.txt           # Python dependencies for the project
└── README.md                  # Project documentation
```

## Setup Instructions

### Prerequisites
- Python 3.7 or higher
- Docker (if using Docker for deployment)

### Installation
1. Clone the repository:
   ```
   git clone <repository-url>
   cd aimechanics-backend
   ```

2. Install the required Python packages:
   ```
   pip install -r requirements.txt
   ```

### Running the Application
To run the FastAPI application locally, use the following command:
```
uvicorn app.deploy_server:app --reload
```

### Using Docker
To build and run the application using Docker, follow these steps:

1. Build the Docker image:
   ```
   docker build -t aimechanics-backend .
   ```

2. Run the Docker container:
   ```
   docker run -d -p 8000:8000 aimechanics-backend
   ```

The application will be accessible at `http://localhost:8000`.

## API Endpoints

### Root Endpoint
- **GET /**: Returns a welcome message.

### Predict Endpoint
- **POST /predict_path**: Classifies audio based on the provided file path.
  - **Form Data**: `file_path` (string) - The path to the audio file.

### Equipment Diagnostic Endpoint
- **POST /equip_diagnostic**: Classifies uploaded audio files.
  - **File**: `file` (audio file) - The audio file to be classified.

## License
This project is licensed under the MIT License. See the LICENSE file for details.