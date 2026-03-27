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

    return file_path


@pytest.fixture
def torch_device_cpu() -> torch.device:
    return torch.device("cpu")
