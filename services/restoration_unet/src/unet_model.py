"""
U-Net Architecture for Image Restoration.

This module defines the foundational U-Net model using TensorFlow and Keras. 
It is designed to ingest 256x256x3 degraded patches and output restored 
256x256x3 patches. It includes skip connections to preserve high-frequency 
spatial details (like film grain and sharp edges).

Usage:
    Run this script directly to test the architecture compilation and 
    verify GPU detection:
    $ python unet_model.py
"""

import tensorflow as tf
from tensorflow.keras import layers, Model
import logging
import os

# Set up the logger (../../../ routes from src/ up to the root revive-ai/logs folder)
log_dir = "../../../logs"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(log_dir, "unet_architecture.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("UNetBuilder")

def conv_block(inputs, num_filters):
    """
    Standard convolutional block for the U-Net Encoder.
    Applies two Conv2D layers with ReLU activation and Same padding.
    """
    x = layers.Conv2D(num_filters, 3, padding="same", activation="relu", kernel_initializer="he_normal")(inputs)
    x = layers.Conv2D(num_filters, 3, padding="same", activation="relu", kernel_initializer="he_normal")(x)
    return x

def encoder_block(inputs, num_filters):
    """
    Encoder block: processes features and pools down the spatial dimensions.
    Returns both the pooled output and the pre-pooled skip connection.
    """
    x = conv_block(inputs, num_filters)
    p = layers.MaxPooling2D((2, 2))(x)
    return x, p

def decoder_block(inputs, skip_features, num_filters):
    """
    Decoder block: upsamples the spatial dimensions and concatenates 
    with the corresponding skip connection from the encoder.
    """
    x = layers.Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(inputs)
    x = layers.Concatenate()([x, skip_features])
    x = conv_block(x, num_filters)
    return x

def build_unet(input_shape=(256, 256, 3)):
    """
    Constructs the complete U-Net architecture.

    Args:
        input_shape (tuple): The spatial dimensions and channels of the input tensors.

    Returns:
        tf.keras.Model: The compiled Keras model ready for training.
    """
    logger.info(f"Initializing U-Net construction with input shape: {input_shape}")
    inputs = layers.Input(shape=input_shape)

    # 1. Encoder (Downsampling)
    s1, p1 = encoder_block(inputs, 64)
    s2, p2 = encoder_block(p1, 128)
    s3, p3 = encoder_block(p2, 256)
    s4, p4 = encoder_block(p3, 512)

    # 2. Bottleneck (Deepest feature representation)
    b1 = conv_block(p4, 1024)

    # 3. Decoder (Upsampling & Skip Connections)
    d1 = decoder_block(b1, s4, 512)
    d2 = decoder_block(d1, s3, 256)
    d3 = decoder_block(d2, s2, 128)
    d4 = decoder_block(d3, s1, 64)

    # 4. Output Layer (3 channels for RGB, sigmoid for 0-1 pixel values)
    outputs = layers.Conv2D(3, 1, padding="same", activation="sigmoid")(d4)

    model = Model(inputs, outputs, name="Restoration_UNet")
    logger.info("U-Net architecture successfully compiled.")
    return model

if __name__ == "__main__":
    # Test 1: Verify Hardware
    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        logger.info(f"SUCCESS: TensorFlow found {len(physical_devices)} GPU(s).")
    else:
        logger.warning("WARNING: No GPU detected. TensorFlow will run on CPU.")

    # Test 2: Build and summarize the model
    try:
        model = build_unet()
        model.summary(print_fn=logger.info)
        
        # Test 3: Save an empty/untrained weights file to verify directory structure
        weights_dir = "../weights"
        os.makedirs(weights_dir, exist_ok=True)
        save_path = os.path.join(weights_dir, "unet_initial.h5")
        model.save(save_path)
        logger.info(f"Initial model successfully saved to {save_path}")
        
    except Exception as e:
        logger.error(f"Failed to build or save U-Net model: {e}")