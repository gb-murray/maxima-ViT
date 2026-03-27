from pathlib import Path

import h5py
import numpy as np
import pytest

from scripts import simulate_data


class _DummyDetector:
    shape = (8, 8)


class _SerialPool:
    def __init__(self, processes=None):
        self.processes = processes

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False

    def imap_unordered(self, func, iterable):
        for item in iterable:
            yield func(item)


@pytest.mark.script_smoke
@pytest.mark.data
def test_generate_dataset_creates_expected_hdf5_layout(tmp_path: Path, monkeypatch):
    def fake_generate_sample(_config):
        image = np.full((8, 8), 2.0, dtype=np.float32)
        label = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], dtype=np.float32)
        return image, label

    monkeypatch.setattr(simulate_data, "get_detector", lambda _alias: _DummyDetector())
    monkeypatch.setattr(simulate_data, "Pool", _SerialPool)
    monkeypatch.setattr(simulate_data, "cpu_count", lambda: 2)
    monkeypatch.setattr(simulate_data, "generate_sample", fake_generate_sample)

    output_path = tmp_path / "sim.hdf5"
    cfg = {
        "paths": {"output_path": str(output_path)},
        "detector": "dummy",
        "generation": {
            "num_images": 4,
            "test_split_ratio": 0.25,
        },
    }

    simulate_data.generate_dataset(cfg)

    assert output_path.exists()
    with h5py.File(output_path, "r") as h5f:
        assert "training_pool" in h5f
        assert "test" in h5f
        assert h5f["training_pool"]["images"].shape == (3, 8, 8)
        assert h5f["training_pool"]["labels"].shape == (3, 6)
        assert h5f["test"]["images"].shape == (1, 8, 8)
        assert h5f["test"]["labels"].shape == (1, 6)
