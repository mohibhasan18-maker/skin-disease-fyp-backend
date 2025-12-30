from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os

# -----------------------------
# App Setup
# -----------------------------
app = FastAPI(title="Skin Disease Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # lock later if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Constants
# -----------------------------
IMAGE_SIZE = 224  # Correct input size for EfficientNetB3
CLASS_NAMES = ["Acne", "Atropic Dermatitis"]

# -----------------------------
# Model paths (safe)
# -----------------------------
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.h5")
WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "weights.h5")

# -----------------------------
# Load Model (ONCE)
# -----------------------------
if not os.path.exists(MODEL_PATH) or not os.path.exists(WEIGHTS_PATH):
    raise FileNotFoundError("Model or weights file not found. Check paths!")

print("Loading model architecture...")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
print("Loading weights...")
model.load_weights(WEIGHTS_PATH)
print("Model loaded successfully.")

# -----------------------------
# Image Preprocessing
# -----------------------------
def preprocess_image(image_bytes: bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE))  # Resize to 224x224

    image_array = np.array(image)
    image_array = tf.keras.applications.efficientnet.preprocess_input(image_array)

    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    return image_array

# -----------------------------
# Routes
# -----------------------------
@app.get("/")
def health_check():
    return {"status": "API is running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    input_tensor = preprocess_image(image_bytes)

    predictions = model.predict(input_tensor)[0]
    predicted_index = int(np.argmax(predictions))
    confidence = float(predictions[predicted_index])

    return {
        "class": CLASS_NAMES[predicted_index],
        "confidence": round(confidence, 4),
        "all_scores": {
            CLASS_NAMES[i]: round(float(predictions[i]), 4)
            for i in range(len(CLASS_NAMES))
        }
    }


# to run the code run this command in terminal:  uvicorn main:app --reload