---
title: Face Similarity Checker
emoji: 🤗
colorFrom: indigo
colorTo: purple
sdk: docker
pinned: false
license: MIT
---

# Face Similarity Checker

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/yolooaster/face-similarity-checker)

A web application to compare two faces and determine their similarity using various deep learning models. This application is built with Python, Flask, and DeepFace, and is ready for deployment on Hugging Face Spaces.

## Features

- **Multiple AI Models:** Choose from a variety of pre-trained face recognition models (VGG-Face, FaceNet, ArcFace, etc.).
- **Customizable Metrics:** Select different distance metrics (Cosine, Euclidean) for similarity calculation.
- **Flexible Face Detection:** Utilize various detector backends (OpenCV, SSD, MTCNN, RetinaFace) for robust face detection.
- **Image Upload:** Upload images directly from your device via drag & drop or file dialog.
- **Webcam Support:** Capture images using your webcam.
- **Image Optimization:** Automatic image optimization for faster processing.
- **Detailed Results:** View verification status, distance score, threshold, and similarity percentage.
- **API Endpoints:** Provides API for programmatic access and dynamic loading of model options.

## Directory Structure

Use code with caution.
Markdown
yolookings-finalproject-machinelearning/
├── README.md # This file
├── Dockerfile # Docker configuration for HF Spaces
├── main.py # Flask application logic
├── run_hf.py # Entrypoint script for Hugging Face
├── pyproject.toml # Project metadata and dependencies
├── uv.lock # Pinned versions of dependencies
├── .python-version # Specifies Python 3.10
├── templates/
│ └── index.html # Frontend HTML
└── .github/
└── workflows/
└── deploy-to-hf.yml # GitHub Action for auto-deployment

## Technology Stack

- **Backend:** Python 3.10, Flask
- **Face Analysis:** DeepFace
- **Frontend:** HTML, CSS, JavaScript (vanilla)
- **Dependency Management & Runner:** uv
- **Containerization:** Docker

## Setup and Installation

### Prerequisites

- Python 3.10 or higher
- `uv` (recommended, install from [astral.sh/uv](https://astral.sh/uv)) or `pip`
- Git

### Local Setup

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yolookings/yolookings-finalproject-machinelearning.git
   cd yolookings-finalproject-machinelearning
   ```

2. **Create and activate a virtual environment:**

   Using `uv` (recommended):

   ```bash
   uv venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

   Or using standard `venv`:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**

   Using `uv` (installs based on `pyproject.toml` and `uv.lock`):

   ```bash
   uv pip install .
   ```

   Alternatively, if you prefer to generate a `requirements.txt`:

   ```bash
   uv pip freeze > requirements.txt
   # Then install with pip:
   # pip install -r requirements.txt
   # Or with uv:
   # uv pip install -r requirements.txt
   ```

   > **Note:** If you need HEIF/HEIC image support, consider adding `pillow-heif` to the `dependencies` array in `pyproject.toml` and re-locking.

## Running the Application

### Locally

Ensure your virtual environment is activated and dependencies are installed.

```bash
python main.py
```

The application will be available at http://localhost:5000.

### Using Docker (Recommended)

Build the Docker image:

```bash
docker build -t face-similarity-app .
```

Run the Docker container:

```bash
docker run -p 7860:7860 face-similarity-app
```

The application will be available at http://localhost:7860.

## API Endpoints

- `GET /`: Serves the main HTML page
- `POST /predict`: Accepts two images and model/metric/detector choices, returns similarity analysis
  - Form data: `image1` (file), `image2` (file), `model` (string), `distance_metric` (string), `detector_backend` (string)
- `GET /api/models`: Returns available models, metrics, detectors, and default choices for the frontend
- `GET /api/health`: Health check endpoint

## Configuration

- **Upload Folder:** `temp_uploads` (created automatically by `main.py` and ensured by `Dockerfile`)
- **Max File Size:** 16MB per image (configurable in `main.py`)
- **Supported Models, Metrics, Detectors:** Defined in `main.py` and dynamically loaded by the frontend via the `/api/models` endpoint

## Deployment to Hugging Face Spaces

This project is configured for deployment to Hugging Face Spaces using Docker. The included GitHub Actions workflow (`.github/workflows/deploy-to-hf.yml`) automatically deploys the application to the `yolooaster/face-similarity-checker` Space when changes are pushed to the main branch.

The Hugging Face Space will use the `Dockerfile` present in this repository. Ensure your `HF_TOKEN` secret is correctly set in your GitHub repository's secrets for the deployment action to function.

## Notes

- The first time DeepFace runs (either locally or in a new Docker container), it will download model weights. This might take a few minutes depending on your internet connection and the models being initialized. Subsequent runs will be faster.
- Face recognition can be memory-intensive. Ensure your local machine or deployment environment has sufficient RAM (e.g., >4GB recommended, more for heavier models or high traffic).
