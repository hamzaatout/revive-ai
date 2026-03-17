"""
DIV2K Dataset Downloader for Revive AI.

This module handles the programmatic downloading and extraction of the 
DIV2K high-resolution validation dataset from ETH Zurich. It logs all 
progress and deletes the residual zip archive to conserve disk space.

Usage:
    Run this script directly from the terminal:
    $ python download_div2k.py
"""

import os
import urllib.request
import zipfile
import shutil
import logging

# Set up the logger
os.makedirs("../../logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("../../logs/data_download.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("DIV2KDownloader")

def download_div2k_scenes():
    """Downloads and extracts the DIV2K high-resolution validation images."""
    url = "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip"
    target_dir = "../../data/raw/restoration_scenes"
    zip_path = os.path.join(target_dir, "DIV2K_valid_HR.zip")
    
    os.makedirs(target_dir, exist_ok=True)

    logger.info("Starting download of DIV2K High-Res Scenes (~350MB)...")
    try:
        urllib.request.urlretrieve(url, zip_path)
        logger.info("Download complete. Extracting files...")
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(target_dir)
            
        logger.info("Extraction complete.")
        
        logger.info("Cleaning up the .zip archive...")
        os.remove(zip_path)
        
        nested_folder = os.path.join(target_dir, "DIV2K_valid_HR")
        if os.path.exists(nested_folder):
            for img_name in os.listdir(nested_folder):
                shutil_src = os.path.join(nested_folder, img_name)
                shutil_dst = os.path.join(target_dir, img_name)
                os.rename(shutil_src, shutil_dst)
            os.rmdir(nested_folder)
            
        logger.info(f"Success! DIV2K scenes are ready in {target_dir}")

    except Exception as e:
        logger.error(f"An error occurred during download or extraction: {e}")

if __name__ == "__main__":
    download_div2k_scenes()