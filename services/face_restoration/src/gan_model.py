"""
Generative Adversarial Network (GAN) for Face Restoration.

This module defines the dual-network architecture required to hallucinate 
and restore missing facial details. It contains:
1. The Generator: Synthesizes high-frequency facial details.
2. The Discriminator: A PatchGAN architecture that penalizes fake/blurry textures.

Usage:
    Run this script directly to compile both networks and verify 
    TensorFlow execution:
    $ python gan_model.py
"""

import tensorflow as tf
from tensorflow.keras import layers, Model
import logging
import os

# Set up the logger
log_dir = "../../../logs"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(log_dir, "gan_architecture.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("FaceGAN")

def downsample(filters, size, apply_batchnorm=True):
    """Standard downsampling block for GANs."""
    initializer = tf.random_normal_initializer(0., 0.02)
    result = tf.keras.Sequential()
    result.add(layers.Conv2D(filters, size, strides=2, padding='same',
                             kernel_initializer=initializer, use_bias=False))
    if apply_batchnorm:
        result.add(layers.BatchNormalization())
    result.add(layers.LeakyReLU())
    return result

def upsample(filters, size, apply_dropout=False):
    """Standard upsampling block for GANs."""
    initializer = tf.random_normal_initializer(0., 0.02)
    result = tf.keras.Sequential()
    result.add(layers.Conv2DTranspose(filters, size, strides=2, padding='same',
                                      kernel_initializer=initializer, use_bias=False))
    result.add(layers.BatchNormalization())
    if apply_dropout:
        result.add(layers.Dropout(0.5))
    result.add(layers.ReLU())
    return result

def build_generator(input_shape=(256, 256, 3)):
    """Builds the Generator (The Art Forger)."""
    inputs = layers.Input(shape=input_shape)

    # Downsampling
    down_stack = [
        downsample(64, 4, apply_batchnorm=False), # (128, 128, 64)
        downsample(128, 4),                       # (64, 64, 128)
        downsample(256, 4),                       # (32, 32, 256)
        downsample(512, 4),                       # (16, 16, 512)
        downsample(512, 4),                       # (8, 8, 512)
        downsample(512, 4),                       # (4, 4, 512)
    ]

    # Upsampling
    up_stack = [
        upsample(512, 4, apply_dropout=True),     # (8, 8, 512)
        upsample(512, 4, apply_dropout=True),     # (16, 16, 512)
        upsample(256, 4),                         # (32, 32, 256)
        upsample(128, 4),                         # (64, 64, 128)
        upsample(64, 4),                          # (128, 128, 64)
    ]

    x = inputs
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)
        
    skips = reversed(skips[:-1])

    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = layers.Concatenate()([x, skip])

    initializer = tf.random_normal_initializer(0., 0.02)
    last = layers.Conv2DTranspose(3, 4, strides=2, padding='same',
                                  kernel_initializer=initializer, activation='tanh')
    
    outputs = last(x)
    return Model(inputs=inputs, outputs=outputs, name="Face_Generator")

def build_discriminator(input_shape=(256, 256, 3)):
    """Builds the PatchGAN Discriminator (The Art Detective)."""
    inp = layers.Input(shape=input_shape, name='target_image')
    tar = layers.Input(shape=input_shape, name='generated_image')
    
    # The discriminator looks at both the input image and the target/generated image
    x = layers.Concatenate()([inp, tar]) # (256, 256, 6)

    down1 = downsample(64, 4, False)(x)  # (128, 128, 64)
    down2 = downsample(128, 4)(down1)    # (64, 64, 128)
    down3 = downsample(256, 4)(down2)    # (32, 32, 256)

    zero_pad1 = layers.ZeroPadding2D()(down3) # (34, 34, 256)
    conv = layers.Conv2D(512, 4, strides=1, kernel_initializer=tf.random_normal_initializer(0., 0.02), use_bias=False)(zero_pad1)
    batchnorm1 = layers.BatchNormalization()(conv)
    leaky_relu = layers.LeakyReLU()(batchnorm1)

    zero_pad2 = layers.ZeroPadding2D()(leaky_relu)
    last = layers.Conv2D(1, 4, strides=1, kernel_initializer=tf.random_normal_initializer(0., 0.02))(zero_pad2)

    return Model(inputs=[inp, tar], outputs=last, name="Face_Discriminator")

if __name__ == "__main__":
    logger.info("Compiling Face GAN Architectures...")
    
    try:
        generator = build_generator()
        discriminator = build_discriminator()
        
        logger.info("--- GENERATOR COMPILED SUCCESSFULLY ---")
        generator.summary(print_fn=logger.info)
        
        logger.info("--- DISCRIMINATOR COMPILED SUCCESSFULLY ---")
        discriminator.summary(print_fn=logger.info)
        
        logger.info("Ready for adversarial training!")
        
    except Exception as e:
        logger.error(f"Failed to build GAN architecture: {e}")