"""
Pre-Trained GFPGAN Face Restoration Worker.

This module acts as the Face Domain microservice. It automatically downloads
the official GFPGANv1.4 weights from Tencent ARC and processes images to 
hallucinate and restore heavily damaged or blurry human faces.

Usage:
    Run this script directly to test the PyTorch inference:
    $ python face_worker.py
"""

import cv2
import os
import urllib.request
import logging
from gfpgan import GFPGANer

# Set up the logger
log_dir = "../../../logs"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(log_dir, "face_worker.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("FaceWorker")

# --- Configuration ---
WEIGHTS_DIR = "../weights"
WEIGHTS_URL = "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth"
WEIGHTS_PATH = os.path.join(WEIGHTS_DIR, "GFPGANv1.4.pth")

TEST_DIR = "../../../data/raw/test_images"
RESULTS_DIR = "../../../data/processed/results"

def download_weights_if_missing():
    """Automatically fetches the pre-trained weights if they aren't downloaded yet."""
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    if not os.path.exists(WEIGHTS_PATH):
        logger.info("Downloading official GFPGANv1.4 weights (This is ~340MB and only happens once)...")
        try:
            urllib.request.urlretrieve(WEIGHTS_URL, WEIGHTS_PATH)
            logger.info("Weights successfully downloaded!")
        except Exception as e:
            logger.error(f"Failed to download weights: {e}")
            raise e
    else:
        logger.info("Pre-trained weights found locally.")

def run_face_restoration():
    logger.info("Initializing Face Worker...")
    
    # 1. Ensure weights exist
    download_weights_if_missing()
    
    os.makedirs(TEST_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # 2. Boot up the GFPGAN Engine
    # We set bg_upsampler=None because our Global U-Net handles the background!
    logger.info("Loading PyTorch GFPGAN model into memory (CPU Mode)...")
    restorer = GFPGANer(
        model_path=WEIGHTS_PATH,
        upscale=2,
        arch='clean',
        channel_multiplier=2,
        bg_upsampler=None 
    )
    
    # 3. Find test images
    test_images = [f for f in os.listdir(TEST_DIR) if f.endswith(('.jpg', '.png', '.jpeg'))]
    if not test_images:
        logger.warning(f"No images found in {TEST_DIR}. Drop a photo with a blurry face in there!")
        return
        
    # 4. Process the images
    for img_name in test_images:
        img_path = os.path.join(TEST_DIR, img_name)
        logger.info(f"Processing faces in {img_name}...")
        
        # Read the image using OpenCV
        img = cv2.imread(img_path)
        
        try:
            # The magic happens here: GFPGAN automatically detects the face, 
            # crops it, restores it, and pastes it back into the original image!
            _, _, restored_img = restorer.enhance(
                img, 
                has_aligned=False, 
                only_center_face=False, 
                paste_back=True
            )
            
            # Save the result
            save_path = os.path.join(RESULTS_DIR, f"face_restored_{img_name}")
            cv2.imwrite(save_path, restored_img)
            logger.info(f"Success! Face restored and saved to {save_path}")
            
        except Exception as e:
            logger.error(f"GFPGAN failed to process {img_name}. Error: {e}")

    logger.info("Face Worker inference complete.")

if __name__ == "__main__":
    run_face_restoration()