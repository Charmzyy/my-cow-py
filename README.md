# FastAPI Service for Breed Classification

This FastAPI service connects your backend application to a machine learning model for breed classification of images. It allows users to submit images through the backend application, which are then processed by the FastAPI service to infer the breed of the cow in the image using the deployed machine learning model.

## Features

- **File Upload Endpoint:** Provides an endpoint `/predict` for uploading images for breed classification.
- **Breed Classification:** Utilizes a pre-trained machine learning model to classify the breed of cows in the uploaded images.
- **Model Inference:** Preprocesses the uploaded images, passes them through the model for inference, and returns the predicted breed along with confidence scores.
- **CORS Middleware:** Includes CORS middleware to handle cross-origin resource sharing, allowing requests from specified origins.

## Dependencies

- FastAPI: Web framework for building APIs with Python.
- TensorFlow: Deep learning framework for training and deploying machine learning models.
- NumPy: Library for numerical computations in Python.
- OpenCV (cv2): Library for computer vision tasks such as image resizing.

## Setup

1. Install dependencies using `pip install -r requirements.txt`.
2. Run the FastAPI service using `uvicorn main:app --host 0.0.0.0 --port 8000`.

## Usage

1. Upload images to the `/predict` endpoint using the backend application <https://github.com/Charmzyy/my-cow-rest>.
2. Receive breed classification results along with confidence scores.
