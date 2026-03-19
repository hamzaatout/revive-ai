"""
CelebA Dataset Downloader for Revive AI.

This module handles the programmatic downloading and extraction of a 
CPU-friendly subset of the CelebA dataset via the Kaggle API. It logs 
all activity and cleans up the massive Kaggle cache immediately after 
extracting the required images.

Usage:
    Run this script directly from the terminal:
    $ python download_celeba.py
"""

import kagglehub
import os
import shutil
import logging

# Resolve paths relative to this script so they are stable regardless of cwd
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir, os.pardir))
LOG_DIR = os.path.join(REPO_ROOT, "logs")

# Set up the logger
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "data_download.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("CelebADownloader")

def download_celeba_subset(num_images=5000):
    """Downloads the CelebA dataset and extracts a specific number of images."""
    logger.info("Starting CelebA dataset download via Kaggle API...")
    
    try:
        cache_path = kagglehub.dataset_download("jessicali9530/celeba-dataset")
        logger.info(f"Dataset temporarily cached at: {cache_path}")
    except Exception as e:
        logger.error(f"Failed to download from Kaggle. Error: {e}")
        return

    target_dir = os.path.join(REPO_ROOT, "data", "raw", "restoration_faces")
    os.makedirs(target_dir, exist_ok=True)

    source_dir = os.path.join(cache_path, "img_align_celeba", "img_align_celeba")
    if not os.path.exists(source_dir):
        source_dir = os.path.join(cache_path, "img_align_celeba")

    if not os.path.isdir(source_dir):
        logger.error(f"Expected image folder not found at: {source_dir}")
        return

    images = [f for f in os.listdir(source_dir) if f.endswith('.jpg')]
    num_to_copy = min(num_images, len(images))
    logger.info(
        f"Found dataset with {len(images)} images. Attempting to extract {num_to_copy} "
        f"images for training (requested {num_images})."
    )

    copied_count = 0
    for i, img_name in enumerate(images[:num_to_copy]):
        src_path = os.path.join(source_dir, img_name)
        dst_path = os.path.join(target_dir, img_name)
        try:
            shutil.copy(src_path, dst_path)
            copied_count += 1
        except Exception as e:
            logger.warning(f"Failed to copy image {img_name}: {e}")

    logger.info(f"Successfully copied {copied_count} images to {target_dir}")

    logger.info("Cleaning up the 200,000+ cached images to save disk space...")
    try:
        shutil.rmtree(cache_path)
        logger.info("Cleanup complete! Cache deleted.")
    except Exception as e:
        logger.warning(f"Could not automatically delete cache. Error: {e}")

if __name__ == "__main__":
    download_celeba_subset()