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


---
## Usage

All operations are controlled via YAML configuration files located in the `src/configs/` directory.

### Data Generation

The `simulate_data.py` script generates a complete HDF5 dataset with training, validation, and test splits, and can upload it directly to a Girder data portal.

1.  Configure dataset parameters in a `configs/data_config.yaml` file.
2.  Set a valid `GIRDER_API_KEY` as an environment variable.
3.  Run the script:
    ```bash
    python scripts/simulate_data.py configs/data_config.yaml
    ```

### Training Workflow

The model is trained using a three-stage, coarse-to-fine process for maximum precision.

#### Head Training 
This quickly trains a custom multi-layer regression head on a large volume of low-resolution (224x224) simulated data.

* **Config**: `configs/training_config.yaml` (set `freeze_backbone: true`, set `learning_rate` ~10e-4)
* **Command**:
    ```bash
    python scripts/train.py configs/training_config.yaml
    ```

#### Stage 2: Full Model Fine-tuning
This fine-tunes the entire network on the same low-resolution data, starting from the Stage 1 checkpoint.

* **Config**: `configs/training_config.yaml` (set `freeze_backbone: false`, set `learning_rate` ~10e-5)
* **Command**:
    ```bash
    python scripts/train.py configs/training_config.yaml --load-checkpoint models/trained_head.pth
    ```

#### Stage 3: High-Resolution Fine-tuning
This final stage adapts the model to high-resolution experimental data for the highest precision.

1.  **Interpolate Weights**: First, create a high-resolution version of your Stage 2 model weights.
    * **Config**: `configs/tuning_config.yaml` (set `image_size`, `hdf5_path` to real data, small `batch_size` and `learning_rate` ~10e-6)
    * **Command**:
        ```bash
        python scripts/interpolate_weights.py configs/tuning_config.yaml models/trained_model.pth models/interpolated_model.pth
        ```

2.  **Run Final Training**: Use the new config and the interpolated checkpoint.
    * **Command**:
        ```bash
        python scripts/train.py configs/tuning_config.yaml --load-checkpoint models/interpolated_model.pth
        ```

### 3. Inference (Calibrating a New Image)

To calibrate a new image using a fully trained model, use the `calibrate.py` script.

```bash
python scripts/calibrate.py --model-path models/best_model.pth --image path/to/my_image.tif --output my_calibration.poni
```

---
## License

This project is licensed under the MIT License. See the `LICENSE` file for details.