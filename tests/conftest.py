import os
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch


# Keep tests from accidentally triggering TensorFlow-side imports via transformers.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")


@pytest.fixture
def tiny_hdf5_file(tmp_path: Path) -> Path:
    file_path = tmp_path / "tiny.hdf5"
    rng = np.random.default_rng(42)

    with h5py.File(file_path, "w") as h5f:
        train = h5f.create_group("train")
        train.create_dataset("images", data=rng.uniform(0, 10, size=(4, 32, 32)).astype(np.float32))
        train.create_dataset("labels", data=rng.uniform(-1, 1, size=(4, 6)).astype(np.float32))

        val = h5f.create_group("validation")
        val.create_dataset("images", data=rng.uniform(0, 10, size=(2, 32, 32)).astype(np.float32))
        val.create_dataset("labels", data=rng.uniform(-1, 1, size=(2, 6)).astype(np.float32))

        normalization = h5f.create_group("normalization")
        normalization.create_dataset("centers", data=np.array([0.15, 0.03, 0.02, 0.0, 0.0, 0.0], dtype=np.float32))
        normalization.create_dataset("scale_factors", data=np.array([0.03, 0.01, 0.01, 0.02, 0.02, 0.0001], dtype=np.float32))

        background = h5f.create_group("background_model")
        background.create_dataset("W", data=rng.uniform(0.1, 2.0, size=(3, 2)).astype(np.float32))
        background.create_dataset("H", data=rng.uniform(0.0, 0.5, size=(2, 32 * 32)).astype(np.float32))
        mask = np.zeros((32, 32), dtype=bool)
        mask[:2, :] = True
        background.create_dataset("master_mask", data=mask)
        background.create_dataset("mask_intensity", data=np.array([4294967295], dtype=np.uint32))

    return file_path


@pytest.fixture
def torch_device_cpu() -> torch.device:
    return torch.device("cpu")
