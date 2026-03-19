"""
Custom Adversarial Training Loop for the Face Restoration GAN.

This module orchestrates the fight between the Generator and Discriminator.
It uses tf.GradientTape to manually calculate losses and update weights.
The Generator tries to minimize its error against the target image while 
simultaneously trying to maximize the Discriminator's error.

Usage:
    Run this script directly from the terminal to begin adversarial training:
    $ python train_gan.py
"""

import tensorflow as tf
import os
import glob
import logging
from gan_model import build_generator, build_discriminator

# Set up the logger
log_dir = "../../../logs"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(log_dir, "gan_training.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("GANTrainer")

# --- Configuration ---
# NOTE: For true production, you should point this ONLY to the CelebA face patches, 
# not the DIV2K scenes. We use the general folder here just to test the plumbing.
INPUTS_DIR = "../../../data/processed/inputs"
TARGETS_DIR = "../../../data/processed/targets"
WEIGHTS_DIR = "../weights"
BATCH_SIZE = 4  # Extremely small batch size for CPU survival
EPOCHS = 2      # Quick dry run

# --- Loss Functions & Optimizers ---
loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def generator_loss(disc_generated_output, gen_output, target):
    """
    Generator wants to fool the discriminator (adversarial loss) AND 
    make the image look exactly like the target (L1 structural loss).
    """
    # 1. Adversarial Loss (Did we fool the detective?)
    gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)
    
    # 2. L1 Loss (Does it match the original pristine face?)
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
    
    # Total G Loss (Multiply L1 by 100 to prioritize structural accuracy)
    total_gen_loss = gan_loss + (100 * l1_loss)
    return total_gen_loss

def discriminator_loss(disc_real_output, disc_generated_output):
    """
    Discriminator wants to perfectly classify real images as 1s and fakes as 0s.
    """
    real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)
    generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)
    return real_loss + generated_loss

# Two separate optimizers for our two networks
generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

# --- Data Loading (Reused from U-Net logic) ---
def load_and_preprocess_image(input_path):
    target_path = tf.strings.regex_replace(input_path, "inputs", "targets")
    
    input_img = tf.cast(tf.image.decode_jpeg(tf.io.read_file(input_path), channels=3), tf.float32) / 255.0
    
    target_img = tf.image.decode_jpeg(tf.io.read_file(target_path), channels=3)
    target_img = tf.image.rgb_to_grayscale(target_img)
    target_img = tf.image.grayscale_to_rgb(target_img)
    target_img = tf.cast(target_img, tf.float32) / 255.0
    
    return input_img, target_img

def build_dataset(input_dir, batch_size):
    file_pattern = os.path.join(input_dir, "*.jpg")
    dataset = tf.data.Dataset.list_files(file_pattern, shuffle=True)
    dataset = dataset.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

# --- The Core Training Step ---
@tf.function # This decorator compiles the function into a highly optimized graph
def train_step(generator, discriminator, input_image, target):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        # 1. The Forger creates a fake face
        gen_output = generator(input_image, training=True)

        # 2. The Detective evaluates both the real target and the fake face
        disc_real_output = discriminator([input_image, target], training=True)
        disc_generated_output = discriminator([input_image, gen_output], training=True)

        # 3. Calculate how wrong both of them were
        gen_total_loss = generator_loss(disc_generated_output, gen_output, target)
        disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

    # 4. Calculate the gradients (The Calculus)
    generator_gradients = gen_tape.gradient(gen_total_loss, generator.trainable_variables)
    discriminator_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    # 5. Update the weights
    generator_optimizer.apply_gradients(zip(generator_gradients, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(discriminator_gradients, discriminator.trainable_variables))
    
    return gen_total_loss, disc_loss

def main():
    logger.info("Initializing GAN Adversarial Training...")
    dataset = build_dataset(INPUTS_DIR, BATCH_SIZE)
    
    generator = build_generator()
    discriminator = build_discriminator()
    
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    
    logger.info("Starting the fight (CPU Mode - This is computationally heavy)...")
    
    for epoch in range(EPOCHS):
        logger.info(f"--- Starting Epoch {epoch + 1}/{EPOCHS} ---")
        
        batch_count = 0
        for input_image, target in dataset:
            gen_loss, disc_loss = train_step(generator, discriminator, input_image, target)
            batch_count += 1
            
            if batch_count % 50 == 0:
                logger.info(f"Batch {batch_count} | Gen Loss: {gen_loss:.4f} | Disc Loss: {disc_loss:.4f}")

        # Save weights after every epoch
        generator.save(os.path.join(WEIGHTS_DIR, "generator_latest.keras"))
        discriminator.save(os.path.join(WEIGHTS_DIR, "discriminator_latest.keras"))
        logger.info(f"Epoch {epoch + 1} complete. Weights saved.")

if __name__ == "__main__":
    main()