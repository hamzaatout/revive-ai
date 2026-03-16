"""
CelebA Dataset Downloader for Revive AI.

This module handles the programmatic downloading and extraction of a 
CPU-friendly subset of the CelebA dataset via the Kaggle API. It is 
specifically designed to prevent local storage overflow by cleaning up 
the massive Kaggle cache immediately after extracting the required images.

Usage:
    Run this script directly from the terminal:
    $ python download_celeba.py
"""

import kagglehub
import os
import shutil

def download_celeba_subset(num_images=5000):
    """
    Downloads the CelebA dataset and extracts a specific number of images.

    This function connects to Kaggle to pull the full dataset, locates the 
    aligned face images, copies the specified subset into the project's 
    raw data directory, and then strictly deletes the Kaggle cache to 
    preserve disk space.

    Args:
        num_images (int, optional): The number of images to extract from 
            the dataset. Defaults to 5000.

    Raises:
        Exception: Prints a warning if the Kaggle cache cannot be 
            automatically deleted due to permission errors.

    Returns:
        None
    """
    print("Downloading CelebA dataset via Kaggle API (This may take a moment)...")
    cache_path = kagglehub.dataset_download("jessicali9530/celeba-dataset")
    print(f"Dataset temporarily cached at: {cache_path}")

    target_dir = "../../data/raw/restoration_faces"
    os.makedirs(target_dir, exist_ok=True)

    source_dir = os.path.join(cache_path, "img_align_celeba", "img_align_celeba")
    if not os.path.exists(source_dir):
        source_dir = os.path.join(cache_path, "img_align_celeba")

    images = [f for f in os.listdir(source_dir) if f.endswith('.jpg')]
    print(f"Extracting {num_images} images for CPU training...")
    
    for i, img_name in enumerate(images[:num_images]):
        src_path = os.path.join(source_dir, img_name)
        dst_path = os.path.join(target_dir, img_name)
        shutil.copy(src_path, dst_path)

    print(f"Successfully copied {num_images} images to {target_dir}")

    print("Cleaning up the 200,000+ cached images to save your hard drive space...")
    try:
        shutil.rmtree(cache_path)
        print("Cleanup complete! Cache deleted.")
    except Exception as e:
        print(f"Warning: Could not automatically delete cache. Error: {e}")

if __name__ == "__main__":
    download_celeba_subset()