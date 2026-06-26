# Max-ViT: Automated 6D Detector Geometry Calibration

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

An end-to-end deep learning model for the rapid and robust determination of 6D detector geometry from 2D powder X-ray diffraction patterns.

## Overview

Traditional methods for calibrating 2D XRD detector geometry rely on heuristic-based algorithms (e.g., edge detection, Hough transforms) that are often brittle and fail on non-ideal experimental data, such as patterns with spotty rings, high texture, or low signal-to-noise. This project overcomes these limitations by using a Vision Transformer (ViT) to learn the complex, non-linear relationship between a diffraction pattern and its underlying geometry.

The result is a fully autonomous system capable of providing fast, precise, and reliable calibrations, suitable for high-throughput experimental workflows.

### Features

* **End-to-End Regression**: Directly predicts the six `pyFAI` geometry parameters from a raw 2D image.
* **Vision Transformer Backbone**: Leverages a ViT's ability to capture global, long-range relationships for superior feature extraction.
* **High-Fidelity Synthetic Data**: Includes a parallelized pipeline for generating large-scale, realistic training datasets with pseudo-Voigt peaks and noise augmentations.
* **Coarse-to-Fine Training**: Implements a three-stage training workflow to achieve maximum precision, starting with coarse training and ending with high-resolution fine-tuning.
* **High-Resolution Support**: Includes a utility to interpolate positional embeddings, adapting pre-trained models to work on high-resolution detector images.

---

## Repository Structure

---

## Setup and Installation

Create and activate a Python 3.11+ environment, then install the project in editable mode:

```bash
python -m pip install --upgrade pip
python -m pip install -e .
```

This installs dependencies from `pyproject.toml` and makes package imports (for example `from maxima_vit.utils import ...`) available from scripts and notebooks without `sys.path` edits.

---

## Docker Workflow

This repository includes a Docker setup that supports:

* Jupyter notebooks
* CLI commands (`maxima-train`, `maxima-calibrate`)
* Live editing of YAML configs and source code without rebuilding

### 1. Build the image

```bash
docker compose build
```

The image is based on a generic Python image (`python:3.11-slim`) and installs:

* PyTorch (`torch==2.8.0`, `torchvision==0.23.0`)
* Project dependencies
* Optional `dev` and `notebook` dependencies
* JupyterLab

### 2. Start JupyterLab

```bash
docker compose up
```

JupyterLab is exposed on `http://localhost:8888`.

Set a custom token if desired:

```bash
JUPYTER_TOKEN=mytoken docker compose up
```

### 3. Run CLI commands in the container

Train:

```bash
docker compose run --rm maxima-vit maxima-train configs/train_swin.yaml
```

Continue from checkpoint:

```bash
docker compose run --rm maxima-vit maxima-train configs/train_swin.yaml --load-checkpoint models/trained_head.pth
```

Calibrate:

```bash
docker compose run --rm maxima-vit maxima-calibrate --model-path models/best_model.pth --config configs/train_swin.yaml --image path/to/my_image.tif --output my_calibration.poni
```

### 4. Editable YAML configs without rebuild

`docker-compose.yml` bind-mounts the repository into `/app` (`./:/app`).
That means files like `configs/train_swin.yaml` are read live from your host filesystem.

You can modify YAML configs (or notebook/code files) locally and rerun commands immediately without rebuilding the image.

---

## Usage

All operations are controlled via YAML configuration files located in the `configs/` directory.

### Training

Run training with the installed CLI entry point:

```bash
maxima-train configs/train_swin.yaml
```

Continue training from a checkpoint:

```bash
maxima-train configs/train_swin.yaml --load-checkpoint models/trained_head.pth
```

### 3. Inference (Calibrating a New Image)

To calibrate a new image using a fully trained model, use the installed CLI entry point.

```bash
maxima-calibrate --model-path models/best_model.pth --config configs/train_swin.yaml --image path/to/my_image.tif --output my_calibration.poni
```

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
