FROM python:3.12-slim

# Set the working directory
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY app/ .

# Copy the svm_ directory
COPY ../svm_/ /app/svm_

# Copy the model file into the container
COPY ../dataset/models/svm_model.joblib /app/models/

# Expose the port the app runs on
EXPOSE 8000

# Command to run the FastAPI application
CMD ["uvicorn", "deploy_server:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]


