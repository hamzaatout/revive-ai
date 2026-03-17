"""
Synthetic Degradation and Patch Extraction Pipeline for Revive AI.

This module processes raw, pristine images by applying a sequence of 
synthetic degradations (grayscale conversion, Gaussian blur, film grain, 
and physical scratches). It then normalizes the dimensions by either 
resizing (for small face images) or extracting a grid of patches 
(for high-resolution scenes) to generate strict 256x256 training pairs.

Usage:
    Run this script directly from the terminal:
    $ python generate_training_pairs.py
"""

import cv2
import numpy as np
import os
import glob
import random
import logging

# Set up the logger
os.makedirs("../../logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("../../logs/data_preprocessing.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("DataPreprocessor")

def add_physical_scratches(img):
    """
    Draws random continuous white and black lines to simulate physical photo damage.
    """
    scratched_img = img.copy()
    h, w = scratched_img.shape[:2]
    num_scratches = random.randint(3, 10)
    
    for _ in range(num_scratches):
        # Random start and end coordinates for the scratch
        x1, y1 = random.randint(0, w), random.randint(0, h)
        x2, y2 = random.randint(0, w), random.randint(0, h)
        
        # Randomly choose if the scratch is white (tear) or black (dirt)
        color = (255, 255, 255) if random.random() > 0.5 else (0, 0, 0)
        thickness = random.randint(1, 3)
        
        cv2.line(scratched_img, (x1, y1), (x2, y2), color, thickness)
        
    return scratched_img

def apply_synthetic_degradation(pristine_img):
    """
    Converts a pristine image to a ruined historical photograph.
    Applies grayscale, blur, noise, and scratches.
    """
    # 1. Strip color (Convert to Grayscale)
    gray = cv2.cvtColor(pristine_img, cv2.COLOR_BGR2GRAY)
    
    # Convert back to 3 channels so dimensions match our network inputs
    degraded = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    
    # 2. Add Gaussian Blur (Out-of-focus camera)
    blur_kernel = random.choice([(3, 3), (5, 5)])
    degraded = cv2.GaussianBlur(degraded, blur_kernel, 0)
    
    # 3. Add Film Grain (Gaussian Noise)
    row, col, ch = degraded.shape
    mean = 0
    sigma = random.uniform(10, 25) # Variance of the noise
    gauss = np.random.normal(mean, sigma, (row, col, ch)).astype(np.float32)
    degraded = cv2.add(degraded.astype(np.float32), gauss)
    degraded = np.clip(degraded, 0, 255).astype(np.uint8)
    
    # 4. Add Scratches
    degraded = add_physical_scratches(degraded)
    
    return degraded

def slice_and_save_patches(pristine_img, degraded_img, base_filename, out_in_dir, out_tar_dir, patch_size=256):
    """
    Slices the full images into 256x256 patches. If the image is smaller 
    than 256x256, it resizes it instead of slicing.
    """
    h, w = pristine_img.shape[:2]
    patch_count = 0
    
    # Logic for CelebA: If image is smaller than patch size, resize it directly
    if h < patch_size or w < patch_size:
        p_patch = cv2.resize(pristine_img, (patch_size, patch_size), interpolation=cv2.INTER_CUBIC)
        d_patch = cv2.resize(degraded_img, (patch_size, patch_size), interpolation=cv2.INTER_CUBIC)
        
        cv2.imwrite(os.path.join(out_in_dir, f"{base_filename}_p0.jpg"), d_patch)
        cv2.imwrite(os.path.join(out_tar_dir, f"{base_filename}_p0.jpg"), p_patch)
        return 1

    # Logic for DIV2K: Slice the massive image into a grid
    for y in range(0, h - patch_size + 1, patch_size):
        for x in range(0, w - patch_size + 1, patch_size):
            p_patch = pristine_img[y:y+patch_size, x:x+patch_size]
            d_patch = degraded_img[y:y+patch_size, x:x+patch_size]
            
            cv2.imwrite(os.path.join(out_in_dir, f"{base_filename}_p{patch_count}.jpg"), d_patch)
            cv2.imwrite(os.path.join(out_tar_dir, f"{base_filename}_p{patch_count}.jpg"), p_patch)
            patch_count += 1
            
    return patch_count

def generate_training_data():
    """Main orchestrator for reading, degrading, slicing, and saving datasets."""
    raw_dirs = [
        "../../data/raw/restoration_faces",
        "../../data/raw/restoration_scenes"
    ]
    
    out_input_dir = "../../data/processed/inputs"
    out_target_dir = "../../data/processed/targets"
    
    os.makedirs(out_input_dir, exist_ok=True)
    os.makedirs(out_target_dir, exist_ok=True)
    
    total_patches = 0
    
    for raw_dir in raw_dirs:
        if not os.path.exists(raw_dir):
            logger.warning(f"Directory not found, skipping: {raw_dir}")
            continue
            
        # Grab all .jpg and .png files
        image_paths = glob.glob(os.path.join(raw_dir, "*.[jp][pn]g"))
        logger.info(f"Found {len(image_paths)} images in {raw_dir}. Starting processing...")
        
        for i, img_path in enumerate(image_paths):
            pristine = cv2.imread(img_path)
            if pristine is None:
                logger.error(f"Failed to read image: {img_path}")
                continue
                
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            
            # Step 1: Ruin the image
            degraded = apply_synthetic_degradation(pristine)
            
            # Step 2: Slice and save both versions
            patches_created = slice_and_save_patches(
                pristine, degraded, base_name, out_input_dir, out_target_dir
            )
            total_patches += patches_created
            
            if (i + 1) % 500 == 0:
                logger.info(f"Processed {i + 1}/{len(image_paths)} images from {os.path.basename(raw_dir)}...")

    logger.info(f"Preprocessing Complete! Successfully generated {total_patches} matched 256x256 patches.")

if __name__ == "__main__":
    generate_training_data()