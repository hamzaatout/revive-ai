"""
Internal Microservice: Global Restoration (U-Net)
Port: 8001
"""
import os
import cv2
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import Response
from contextlib import asynccontextmanager

WEIGHTS_PATH = "../weights/best_unet_model.keras"
model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    try:
        model = tf.keras.models.load_model(WEIGHTS_PATH)
        print("U-Net loaded successfully.")
    except Exception as e:
        print(f"Error loading U-Net: {e}")
    yield

app = FastAPI(title="Internal U-Net Service", lifespan=lifespan)

@app.post("/restore_global")
async def restore_global(file: UploadFile = File(...)):
    if model is None: raise HTTPException(status_code=503)
    
    image_bytes = await file.read()
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (256, 256), interpolation=cv2.INTER_CUBIC)
    input_tensor = np.expand_dims(img_resized.astype(np.float32) / 255.0, axis=0)
    
    prediction = model.predict(input_tensor, verbose=0)
    
    output_array = np.squeeze(prediction)
    output_img = (output_array * 255.0).astype(np.uint8)
    output_bgr = cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR)
    
    success, encoded_image = cv2.imencode('.jpg', output_bgr)
    return Response(content=encoded_image.tobytes(), media_type="image/jpeg")