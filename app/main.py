from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import numpy as np
import cv2

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:8001",
    "https://my-cow-rest.onrender.com",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# MODEL = tf.keras.models.load_model(r"C:\Users\User\Desktop\final\my-cow-py\my_model.h5")

MODEL = tf.keras.models.load_model(r"/app/my_model.h5")


CLASS_NAMES = ['Ayrshire', 'Black-Angus', 'Boran', 'Guernsey', 'Holstein', 'Jersey', 'Sahiwal', 'Zebu']


def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image


def resize_image(image, target_size=(256, 256)):
    resized_image = cv2.resize(image, target_size)
    return resized_image




@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = read_file_as_image(await file.read())
    resized_image = resize_image(image, target_size=(256, 256))
    img_batch = np.expand_dims(resized_image, 0)
    predictions = MODEL.predict(img_batch)

    print("Predictions array:", predictions)  # Print the predictions array

    predicted_class_index = np.argmax(predictions[0])
    print("Predicted class index:", predicted_class_index)  # Print the predicted class index

    predicted_class = CLASS_NAMES[predicted_class_index]
    confidence = float(np.max(predictions[0])) * 100

    if confidence < 48:
        return {
            'class': 'Not a cow',
            'confidence': confidence
        }
    else:
        return {
            'class': predicted_class,
            'confidence': confidence
        }
    