"""
FastAPI Microservice for the Restoration U-Net.

This module exposes the trained TensorFlow model as a REST API. 
It accepts an uploaded image, processes it in memory, runs the 
restoration inference, and returns the cleaned image back to the client.

Usage:
    Run the server using Uvicorn from the terminal:
    $ uvicorn app:app --reload
"""

import io
import cv2
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import Response
from contextlib import asynccontextmanager
import logging

# Set up basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("RestorationAPI")

# Global variable to hold our model in memory
model = None
WEIGHTS_PATH = "../weights/best_unet_model.keras"

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager: Loads the model into memory exactly once on startup."""
    global model
    logger.info("Booting up Restoration API...")
    try:
        model = tf.keras.models.load_model(WEIGHTS_PATH)
        logger.info("U-Net model successfully loaded into memory!")
    except Exception as e:
        logger.error(f"Failed to load model from {WEIGHTS_PATH}. Error: {e}")
    yield
    # Cleanup code would go here if we needed it on shutdown
    logger.info("Shutting down API...")

# Initialize FastAPI
app = FastAPI(
    title="Revive AI - Restoration Worker",
    description="Microservice for removing physical damage from historical photographs.",
    version="1.0.0",
    lifespan=lifespan
)

def process_image_tensor(image_bytes: bytes):
    """Converts uploaded bytes into the 256x256 normalized tensor the U-Net expects."""
    # Convert raw bytes to an OpenCV image array
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        raise ValueError("Invalid image file format.")
        
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, (256, 256), interpolation=cv2.INTER_CUBIC)
    
    # Normalize to [0, 1] and add batch dimension
    img_normalized = img_resized.astype(np.float32) / 255.0
    img_tensor = np.expand_dims(img_normalized, axis=0)
    
    return img_tensor

def create_response_image(prediction_tensor):
    """Converts the neural network's output tensor back into a JPEG byte stream."""
    output_array = np.squeeze(prediction_tensor)
    output_array = (output_array * 255.0).astype(np.uint8)
    
    # Convert RGB back to BGR for OpenCV encoding
    output_img = cv2.cvtColor(output_array, cv2.COLOR_RGB2BGR)
    
    # Encode the image to JPEG format in memory
    success, encoded_image = cv2.imencode('.jpg', output_img)
    if not success:
        raise ValueError("Failed to encode the processed image.")
        
    return encoded_image.tobytes()

@app.post("/restore")
async def restore_image(file: UploadFile = File(...)):
    """
    Endpoint: Accepts an image file and returns the restored version.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model is currently unavailable.")
        
    try:
        # 1. Read the uploaded file
        image_bytes = await file.read()
        
        # 2. Preprocess mathematically
        input_tensor = process_image_tensor(image_bytes)
        
        # 3. Predict (The heavy lifting)
        prediction = model.predict(input_tensor, verbose=0)
        
        # 4. Postprocess back to standard image bytes
        result_bytes = create_response_image(prediction)
        
        # 5. Send it straight back to the user's browser/frontend
        return Response(content=result_bytes, media_type="image/jpeg")
        
    except Exception as e:
        logger.error(f"Error processing request: {e}")
        raise HTTPException(status_code=400, detail="Failed to process image.")