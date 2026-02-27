# Revive

**Restoring and Colorizing Damaged Historical Photographs with AI**

## Objective

Revive is a university neural networks project that develops an AI-based system to restore and colorize damaged black-and-white historical photographs. The system leverages deep learning techniques to repair visual degradation (scratches, fading, noise) and apply historically accurate colorization — preserving the visual integrity and cultural authenticity of the original images.

## Directory Structure

```
revive-ai/
├── data/
│   ├── raw/          # Original, unprocessed image datasets
│   └── processed/    # Cleaned and pre-processed images ready for training
├── notebooks/        # Jupyter notebooks for experimentation and visualization
├── src/              # Core source code (models, training loops, utilities)
├── models/           # Saved model checkpoints and weights
├── scripts/          # Standalone scripts for training, evaluation, and inference
├── requirements.txt
└── README.md
```

## Datasets

The project makes use of the following publicly available datasets:

| Dataset | Description |
|---|---|
| **[DeOldify](https://github.com/jantic/DeOldify)** | A deep learning based project for colorizing and restoring old images and video |
| **Historical Color Image Dataset** | A curated collection of digitized historical photographs with ground-truth color references |
| **[Flickr Commons](https://www.flickr.com/commons)** | A large public archive of historical images contributed by cultural institutions worldwide |

## Setup and Installation

### Prerequisites

- Python 3.8+
- [pip](https://pip.pypa.io/en/stable/) or [conda](https://docs.conda.io/)
- A CUDA-capable GPU is recommended for training

### 1. Clone the repository

```bash
git clone https://github.com/hamzaatout/revive-ai.git
cd revive-ai
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
# On Linux/macOS
source venv/bin/activate
# On Windows
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download datasets

Place raw image data inside `data/raw/`. Refer to each dataset's official documentation for download instructions.

### 5. Run a notebook

```bash
jupyter notebook notebooks/
```

## Team Members

| Name | Role |
|---|---|
| **Hamza Atout** | Project Lead & ML Engineer |
| **Farah Abdallah** | Data Engineer & Model Training |
| **Nour Hamieh** | Computer Vision & Evaluation |
