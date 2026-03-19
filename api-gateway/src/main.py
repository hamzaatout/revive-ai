"""
API Gateway Orchestrator
Port: 8000 (Exposed to Frontend)

This service acts as the central router for the Revive AI frontend. 
It receives an image, passes it to the U-Net microservice for global 
scratch removal, and then passes that cleaned image to the GFPGAN 
microservice for face hallucination, before returning the final result.
"""
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
import httpx

app = FastAPI(title="Revive AI - API Gateway")

# Allow the frontend (HTML/JS) to communicate with this Gateway
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, restrict this to your actual domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Internal Microservice Addresses
UNET_SERVICE_URL = "http://127.0.0.1:8001/restore_global"
GFPGAN_SERVICE_URL = "http://127.0.0.1:8002/restore_face"

@app.post("/process_image")
async def process_image(file: UploadFile = File(...)):
    """
    Main orchestration endpoint. Chains the U-Net and GFPGAN models sequentially.
    """
    image_bytes = await file.read()

    # We use a 60-second timeout because running PyTorch/TensorFlow 
    # simultaneously on a CPU requires significant processing time.
    async with httpx.AsyncClient(timeout=60.0) as client:
        
        # --- STEP 1: Global Restoration (U-Net) ---
        try:
            unet_response = await client.post(
                UNET_SERVICE_URL,
                files={"file": ("image.jpg", image_bytes, "image/jpeg")}
            )
            unet_response.raise_for_status()
            unet_cleaned_bytes = unet_response.content
        except httpx.RequestError as e:
            raise HTTPException(status_code=500, detail=f"U-Net Network Error: {e}")
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=500, detail=f"U-Net Processing Error: {e.response.text}")

        # --- STEP 2: Face Hallucination (GFPGAN) ---
        # We pass the U-Net result directly to the face worker
        try:
            gfpgan_response = await client.post(
                GFPGAN_SERVICE_URL,
                files={"file": ("image.jpg", unet_cleaned_bytes, "image/jpeg")}
            )
            gfpgan_response.raise_for_status()
            final_bytes = gfpgan_response.content
        except httpx.RequestError as e:
            raise HTTPException(status_code=500, detail=f"GFPGAN Network Error: {e}")
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=500, detail=f"GFPGAN Processing Error: {e.response.text}")

    # --- STEP 3: Return the final masterpiece to the browser ---
    return Response(content=final_bytes, media_type="image/jpeg")