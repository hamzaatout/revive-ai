"""
Training Pipeline for the Restoration U-Net.

This module handles the end-to-end training process. It creates an efficient 
data pipeline to load paired image patches (degraded inputs and pristine targets) 
from the hard drive, compiles the U-Net architecture, and trains the model 
using Mean Absolute Error (L1 Loss) to ensure sharp pixel restoration.

Usage:
    Run this script directly from the terminal to begin training:
    $ python train.py
"""

import tensorflow as tf
import os
import glob
import logging
from unet_model import build_unet

# Set up the logger
log_dir = "../../../logs"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(log_dir, "training.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("UNetTrainer")

# --- Configuration ---
INPUTS_DIR = "../../../data/processed/inputs"
TARGETS_DIR = "../../../data/processed/targets"
WEIGHTS_DIR = "../weights"
BATCH_SIZE = 8  # Kept small for CPU memory safety
EPOCHS = 5      # Short run to verify the pipeline works

def load_and_preprocess_image(input_path):
    """
    Reads an image pair from disk and normalizes pixel values to [0, 1].
    The target is dynamically converted to grayscale to isolate the restoration task.
    """
    # Generate the corresponding target path
    target_path = tf.strings.regex_replace(input_path, "inputs", "targets")
    
    # Process Input (Degraded)
    input_img = tf.io.read_file(input_path)
    input_img = tf.image.decode_jpeg(input_img, channels=3)
    input_img = tf.cast(input_img, tf.float32) / 255.0
    
    # Process Target (Pristine)
    target_img = tf.io.read_file(target_path)
    target_img = tf.image.decode_jpeg(target_img, channels=3)
    
    # NEW FIX: Strip color from target so the model only learns structural restoration
    target_img = tf.image.rgb_to_grayscale(target_img)
    target_img = tf.image.grayscale_to_rgb(target_img) # Back to 3 channels for U-Net
    
    target_img = tf.cast(target_img, tf.float32) / 255.0
    
    return input_img, target_img

def build_dataset(input_dir, batch_size):
    """Constructs an efficient tf.data pipeline for image pairs."""
    # Grab all .jpg files in the inputs directory
    file_pattern = os.path.join(input_dir, "*.jpg")
    dataset = tf.data.Dataset.list_files(file_pattern, shuffle=True)
    
    # Map the preprocessing function across the dataset
    dataset = dataset.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Batch and prefetch for performance
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

def main():
    logger.info("Initializing Training Pipeline...")
    
    # 1. Verify Data Exists
    input_files = glob.glob(os.path.join(INPUTS_DIR, "*.jpg"))
    if not input_files:
        logger.error(f"No training data found in {INPUTS_DIR}. Did you run generate_training_pairs.py?")
        return
    logger.info(f"Found {len(input_files)} training patches.")
    
    # 2. Build the Conveyor Belt
    logger.info("Building data pipeline...")
    train_dataset = build_dataset(INPUTS_DIR, BATCH_SIZE)
    
    # 3. Compile the Brain
    logger.info("Compiling U-Net model...")
    model = build_unet(input_shape=(256, 256, 3))
    
    # We use Mean Absolute Error (MAE) instead of Mean Squared Error (MSE) 
    # because MAE produces much sharper images in restoration tasks.
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                  loss="mae", 
                  metrics=["accuracy"])
    
    # 4. Set up Callbacks (Save the best version automatically)
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    checkpoint_path = os.path.join(WEIGHTS_DIR, "best_unet_model.keras")
    
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            save_best_only=True,
            monitor="loss",
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="loss", 
            patience=2, 
            restore_best_weights=True
        )
    ]
    
    # 5. Start Training
    logger.info("Starting training loop (CPU Mode - This will take some time)...")
    try:
        history = model.fit(
            train_dataset,
            epochs=EPOCHS,
            callbacks=callbacks
        )
        logger.info(f"Training complete! Best weights saved to {checkpoint_path}")
    except KeyboardInterrupt:
        logger.warning("Training manually interrupted by user. Saving current state is recommended.")
    except Exception as e:
        logger.error(f"A fatal error occurred during training: {e}")

if __name__ == "__main__":
    main()