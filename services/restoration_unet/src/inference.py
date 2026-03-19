"""
Inference Pipeline for the Restoration U-Net.

This module loads the trained U-Net weights and processes new, unseen 
damaged photographs. It handles the mathematical normalization required 
for the neural network, executes the prediction, and reconstructs the 
output tensor back into a standard, viewable image file.

Usage:
    Run this script directly from the terminal. By default, it will look 
    for images in a 'test_images' folder and save the restored versions 
    to a 'results' folder.
    $ python inference.py
"""

import os
import cv2
import numpy as np
import tensorflow as tf
import logging

# Set up the logger
log_dir = "../../../logs"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(log_dir, "inference.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("UNetInference")

# --- Configuration ---
WEIGHTS_PATH = "../weights/best_unet_model.keras"
TEST_DIR = "../../../data/raw/test_images"
RESULTS_DIR = "../../../data/processed/results"

def preprocess_image(img_path, target_size=(256, 256)):
    """Loads an image, resizes it to match the U-Net input, and normalizes it."""
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Could not read image at {img_path}")
        
    # OpenCV loads in BGR, but we trained on RGB (via tf.image.decode_jpeg)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize to the 256x256 tensor shape the network expects
    img_resized = cv2.resize(img, target_size, interpolation=cv2.INTER_CUBIC)
    
    # Normalize to [0, 1]
    img_normalized = img_resized.astype(np.float32) / 255.0
    
    # Add batch dimension: (256, 256, 3) -> (1, 256, 256, 3)
    img_tensor = np.expand_dims(img_normalized, axis=0)
    
    return img, img_tensor

def postprocess_tensor(tensor):
    """Converts the neural network's raw output tensor back into a viewable image."""
    # Remove the batch dimension: (1, 256, 256, 3) -> (256, 256, 3)
    output_array = np.squeeze(tensor)
    
    # Denormalize from [0, 1] back to [0, 255]
    output_array = (output_array * 255.0).astype(np.uint8)
    
    # Convert back from RGB to BGR so OpenCV can save it correctly
    output_img = cv2.cvtColor(output_array, cv2.COLOR_RGB2BGR)
    
    return output_img

def run_inference():
    """Main orchestrator for loading the model and predicting on test images."""
    logger.info("Initializing Inference Engine...")
    
    if not os.path.exists(WEIGHTS_PATH):
        logger.error(f"No trained weights found at {WEIGHTS_PATH}. Did training complete?")
        return
        
    os.makedirs(TEST_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # 1. Load the Brain
    logger.info("Loading U-Net weights into memory...")
    try:
        model = tf.keras.models.load_model(WEIGHTS_PATH)
    except Exception as e:
        logger.error(f"Failed to load model. Error: {e}")
        return
        
    # 2. Find Test Images
    test_images = [f for f in os.listdir(TEST_DIR) if f.endswith(('.jpg', '.png', '.jpeg'))]
    if not test_images:
        logger.warning(f"No test images found in {TEST_DIR}. Please drop a damaged photo in there and run again.")
        return
        
    logger.info(f"Found {len(test_images)} images to restore.")
    
    # 3. Execute Predictions
    for img_name in test_images:
        img_path = os.path.join(TEST_DIR, img_name)
        logger.info(f"Processing {img_name}...")
        
        try:
            # Prepare
            original_img, input_tensor = preprocess_image(img_path)
            
            # Predict (This is where the math happens)
            prediction_tensor = model.predict(input_tensor, verbose=0)
            
            # Reconstruct
            restored_img = postprocess_tensor(prediction_tensor)
            
            # Save the result side-by-side for easy comparison
            # We resize the original to 256x256 just so it visually matches the output scale
            original_resized = cv2.resize(original_img, (256, 256))
            comparison = np.hstack((original_resized, restored_img))
            
            save_path = os.path.join(RESULTS_DIR, f"restored_{img_name}")
            cv2.imwrite(save_path, comparison)
            logger.info(f"Success! Saved comparison to {save_path}")
            
        except Exception as e:
            logger.error(f"Failed to process {img_name}. Error: {e}")

    logger.info("Inference complete.")

if __name__ == "__main__":
    run_inference()