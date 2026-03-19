"""
Internal Microservice: Face Hallucination (GFPGAN)
Port: 8002
"""
# --- MONKEY PATCH FOR BASICSR BUG ---
# This intercepts the bad import and redirects it to the modern PyTorch location
import sys
import torchvision.transforms.functional as tv_f
sys.modules['torchvision.transforms.functional_tensor'] = tv_f
# ------------------------------------

import os
import cv2
import numpy as np
import urllib.request
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import Response
from contextlib import asynccontextmanager
from gfpgan import GFPGANer

WEIGHTS_DIR = "../weights"
WEIGHTS_PATH = os.path.join(WEIGHTS_DIR, "GFPGANv1.4.pth")
WEIGHTS_URL = "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth"

face_restorer = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global face_restorer
    
    # Self-healing: Download weights if they don't exist
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    if not os.path.exists(WEIGHTS_PATH):
        print("Downloading official GFPGANv1.4 weights (This happens once)...")
        try:
            urllib.request.urlretrieve(WEIGHTS_URL, WEIGHTS_PATH)
            print("Weights successfully downloaded!")
        except Exception as e:
            print(f"Failed to download weights: {e}")
            
    try:
        face_restorer = GFPGANer(
            model_path=WEIGHTS_PATH, upscale=2, arch='clean', 
            channel_multiplier=2, bg_upsampler=None
        )
        print("GFPGAN loaded successfully.")
    except Exception as e:
        print(f"Error loading GFPGAN: {e}")
    yield

app = FastAPI(title="Internal GFPGAN Service", lifespan=lifespan)

@app.post("/restore_face")
async def restore_face(file: UploadFile = File(...)):
    if face_restorer is None: raise HTTPException(status_code=503, detail="Model not loaded")
    
    image_bytes = await file.read()
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    _, _, restored_img = face_restorer.enhance(
        img, has_aligned=False, only_center_face=False, paste_back=True
    )
    
    success, encoded_image = cv2.imencode('.jpg', restored_img)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to encode image")
        
    return Response(content=encoded_image.tobytes(), media_type="image/jpeg")