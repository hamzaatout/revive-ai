# Historical Photo Restoration & Colorization Pipeline

[![Course](https://img.shields.io/badge/Course-EECE%20693%3A%20Neural%20Networks-blue.svg)]()
[![Institution](https://img.shields.io/badge/Institution-American%20University%20of%20Beirut%20(AUB)-maroon.svg)]()
[![Architecture](https://img.shields.io/badge/Architecture-Microservices%20%7C%20Docker-orange.svg)]()
[![Framework](https://img.shields.io/badge/Framework-TensorFlow%20%7C%20Keras-orange.svg)]()

A production-grade, distributed deep learning pipeline designed to automate the restoration and colorization of heavily damaged historical black-and-white photographs.

## 👥 Team Members

* **Hamza Atout** - [hsa60@mail.aub.edu](mailto:hsa60@mail.aub.edu)
* **Nour Hammieh** - [nnh30@mail.aub.edu](mailto:nnh30@mail.aub.edu)
* **Farah Abdallah** - [fia18@mail.aub.edu](mailto:fia18@mail.aub.edu)

---

## 📑 Table of Contents

1. [Project Overview](#-project-overview)
2. [System Architecture](#-system-architecture)
3. [Models & Datasets](#-models--datasets)
4. [Evaluation Metrics](#-evaluation-metrics)
5. [Complete Project Structure](#-complete-project-structure)
6. [Getting Started (CPU-Only Local Development)](#-getting-started-cpu-only-local-development)

---

## 🔬 Project Overview

This project addresses the preservation of historical visual artifacts by splitting the problem into two isolated domains: **Structural Restoration** (fixing scratches, tears, and film grain) and **Semantic Colorization** (predicting historically plausible RGB values).

Unlike standard monolithic Jupyter Notebook scripts, this system is built as a **Dockerized Microservices Architecture**. It features an intelligent **API Gateway** that acts as an orchestrator, utilizing conditional routing (Variance of Laplacian) to dynamically deploy heavy facial-hallucination GANs only when blurry faces are detected, thereby drastically optimizing computational resources.

---

## 🏗️ System Architecture

The pipeline follows a strict, stateless microservices pattern:

1. **API Gateway (The Orchestrator):** Receives the raw image, standardizes it to a normalized grayscale tensor, and orchestrates asynchronous HTTP calls to internal workers.
2. **Global Restoration Worker:** Strips physical scratches and noise across the entire photo using baseline structural models.
3. **Conditional Routing (Face Prior):** The Gateway scans the cleaned image. If a blurry face is detected (via low Laplacian variance), the cropped face is sent to an isolated **GFP-GAN Worker** (TensorFlow port) for high-res hallucination, then seamlessly blended back.
4. **Colorization Worker:** The fully restored, sharp black-and-white image is routed to the color prediction models to map semantic RGB values.

---

## 🧠 Models & Datasets

### Phase 1: Structural Restoration

* **Models:**
  * **Custom U-Net (TensorFlow Baseline):** Built using `tf.keras` and trained locally on extracted patches to establish a core mathematical baseline.
  * **MS-OldPhoto (Bringing Old Photos Back to Life):** Dual-VAE architecture for robust global scratch removal (adapted for CPU inference).
  * **SwinIR / Autoencoders:** Custom TensorFlow implementations for advanced denoising.
  * **GFP-GAN:** Generative Facial Prior GAN for conditional facial hallucination.

* **Datasets:** CelebA (aligned faces), DIV2K (complex scenes). *Synthetically degraded using custom offline OpenCV scripts.*

### Phase 2: Semantic Colorization

* **Models:**
  * **DeOldify-inspired TF Model:** Self-attention architectures adapted for TensorFlow to achieve stable colorization.
  * **Custom cGAN (Pix2Pix Variant):** Developed from scratch using TensorFlow/Keras to minimize perceptual loss.
  * **InstColorization:** Instance-aware colorization for complex scenes.

* **Datasets:** ImageNet (color/grayscale pairs), Flickr Commons (historical archives), Historical Color Image Dataset.

---

## 📊 Evaluation Metrics

To rigorously evaluate model performance, we employ both structural and perceptual metrics:

* **PSNR (Peak Signal-to-Noise Ratio):** Measures absolute pixel-level reconstruction.
* **SSIM (Structural Similarity Index):** Evaluates structural and contrast degradation.
* **LPIPS (Learned Perceptual Image Patch Similarity):** Measures deep-feature perceptual similarity (closer to human vision).
* **FID (Fréchet Inception Distance):** Evaluates the realism of GAN-generated textures and colors.

---

## 📂 Complete Project Structure

The repository is strictly modularized to prevent dependency conflicts between different models.

```text
revive-ai/
├── .github/                           # CI/CD workflows (GitHub Actions)
│   └── workflows/
│       ├── test-gateway.yml
│       └── build-dockers.yml
├── docs/                              # Comprehensive Documentation
│   ├── architecture/                  # Architecture diagrams (Mermaid.js/PNGs)
│   ├── api-specs/                     # OpenAPI (Swagger) YAML specifications
│   ├── model-training/                # Notes on training hyper-parameters
│   └── setup-guide.md                 # Detailed local setup instructions
├── api-gateway/                       # 🚦 The Orchestrator (FastAPI)
│   ├── src/
│   │   ├── main.py                    # Async routing & conditional logic
│   │   ├── schemas.py                 # Pydantic models for API validation
│   │   └── utils/
│   │       ├── blur_detection.py      # Variance of Laplacian CV2 script
│   │       └── image_processing.py    # Grayscale standardization
│   ├── tests/                         # Gateway unit tests
│   ├── Dockerfile
│   └── requirements.txt
├── services/                          # 🧠 ML Workers (Isolated Environments)
│   ├── restoration_unet/              # Baseline Restoration Service
│   │   ├── src/
│   │   │   ├── main.py
│   │   │   └── unet_model.py
│   │   ├── weights/                   # TF SavedModel/.h5 files (IGNORED IN GIT)
│   │   ├── Dockerfile                 # Configured for CPU-only TensorFlow
│   │   └── requirements.txt
│   ├── restoration_gfpgan/            # Conditional Face Hallucination Service
│   │   ├── src/
│   │   │   ├── main.py
│   │   │   └── gfpgan_wrapper.py
│   │   ├── weights/                   # TF SavedModel/.pb files
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   ├── colorization_baseline/         # Baseline Colorization Service
│   │   ├── src/
│   │   │   ├── main.py
│   │   │   └── color_model_wrapper.py
│   │   ├── weights/
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   └── colorization_custom_cgan/      # Custom Colorization Service
│       ├── src/
│       │   ├── main.py
│       │   └── tf_cgan_model.py
│       ├── weights/
│       ├── Dockerfile
│       └── requirements.txt
├── scripts/                           # 🛠️ Utility & Data Pipelines
│   ├── data_preprocessing/
│   │   ├── synthetic_degradation.py   # Script to add scratches/noise/sepia
│   │   └── patch_extractor.py         # Slices images into 256x256 patches
│   └── download_weights.sh            # Pulls pre-trained weights from cloud storage
├── frontend/                          # 🌐 Web UI / Internal Research Hub
│   ├── public/
│   │   ├── index.html
│   │   ├── style.css
│   │   └── script.js                  # Handles Base64 image payload to Gateway
│   └── Dockerfile                     # Nginx static server
├── .gitignore                         # Ignores /weights, __pycache__, .env, etc.
├── docker-compose.yml                 # Master orchestrator for local testing
└── README.md                          # You are here