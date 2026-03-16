"""
DIV2K Dataset Downloader for Revive AI.

This module handles the programmatic downloading and extraction of the 
DIV2K high-resolution validation dataset from ETH Zurich. It streams the 
download to minimize RAM usage, extracts the images into the project's 
raw data directory, and deletes the residual zip archive to conserve 
disk space.

Usage:
    Run this script directly from the terminal:
    $ python download_div2k.py
"""

import os
import urllib.request
import zipfile

def download_div2k_scenes():
    """
    Downloads and extracts the DIV2K high-resolution validation images.

    This function connects to the official ETH Zurich vision server, 
    downloads the DIV2K_valid_HR.zip file in chunks, extracts the 
    images to the local restoration_scenes directory, and cleans up 
    the zip file afterward.

    Returns:
        None
    """
    url = "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip"
    target_dir = "../../data/raw/restoration_scenes"
    zip_path = os.path.join(target_dir, "DIV2K_valid_HR.zip")
    
    os.makedirs(target_dir, exist_ok=True)

    print("Downloading DIV2K High-Res Scenes (This is ~350MB)...")
    try:
        # Download the file
        urllib.request.urlretrieve(url, zip_path)
        print("Download complete. Extracting files...")
        
        # Extract the contents
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # The zip file contains a folder named 'DIV2K_valid_HR'
            zip_ref.extractall(target_dir)
            
        print("Extraction complete.")
        
        # Cleanup the heavy zip file
        print("Cleaning up the .zip archive...")
        os.remove(zip_path)
        
        # Optional: Move images out of the nested folder to the parent target_dir
        nested_folder = os.path.join(target_dir, "DIV2K_valid_HR")
        if os.path.exists(nested_folder):
            for img_name in os.listdir(nested_folder):
                shutil_src = os.path.join(nested_folder, img_name)
                shutil_dst = os.path.join(target_dir, img_name)
                os.rename(shutil_src, shutil_dst)
            os.rmdir(nested_folder)
            
        print(f"Success! DIV2K scenes are ready in {target_dir}")

    except Exception as e:
        print(f"An error occurred during download or extraction: {e}")

if __name__ == "__main__":
    import shutil # Imported here for the nested folder cleanup
    download_div2k_scenes()