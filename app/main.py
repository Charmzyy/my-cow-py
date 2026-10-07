import json
import os
from io import BytesIO

import cv2
import keras
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError

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

# Model + metadata produced by CowBreedClassification_v2.ipynb (sections 12-13).
# model_meta.json holds the class order, input size and threshold the model was trained/tuned with.
MODEL_PATH = os.getenv("MODEL_PATH", "/app/cow_breed_v2.keras")
META_PATH = os.getenv("MODEL_META_PATH", "/app/model_meta.json")

with open(META_PATH) as fh:
    META = json.load(fh)

MODEL = keras.models.load_model(MODEL_PATH)
CLASS_NAMES = META["class_names"]
IMG_SIZE = int(META["img_size"])
THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", META.get("threshold", 0.6)))  # 0-1


def read_file_as_image(data) -> np.ndarray:
    # .convert("RGB"): PNGs with transparency (4 channels) and greyscale photos (1 channel)
    # would otherwise crash the model, which expects exactly 3 channels
    return np.array(Image.open(BytesIO(data)).convert("RGB"))


def resize_image(image, target_size):
    return cv2.resize(image, target_size)


@app.get("/health")
def health():
    return {"status": "ok", "classes": len(CLASS_NAMES), "threshold": THRESHOLD}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image = read_file_as_image(await file.read())
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=422, detail="File is not a readable image")

    # Raw 0-255 RGB, no /255: the model was trained this way (EfficientNet rescales internally)
    resized_image = resize_image(image, (IMG_SIZE, IMG_SIZE)).astype("float32")
    img_batch = np.expand_dims(resized_image, 0)
    predictions = MODEL.predict(img_batch, verbose=0)[0]

    predicted_class_index = int(np.argmax(predictions))
    predicted_class = CLASS_NAMES[predicted_class_index]
    confidence = float(predictions[predicted_class_index]) * 100

    # A trained "not_cow" class, or a low-confidence guess, both mean "Not a cow"
    if predicted_class == "not_cow" or confidence < THRESHOLD * 100:
        return {"class": "Not a cow", "confidence": confidence}
    return {"class": predicted_class, "confidence": confidence}
