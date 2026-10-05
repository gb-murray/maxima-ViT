from pathlib import Path

import numpy as np
import torch

from xrd_geometry.checkpoints import read_checkpoint
from xrd_geometry.cli import calibrate as calibrate_cli
from xrd_geometry.cli import train as train_cli
from xrd_geometry.models import DeterministicHead, FullCovarianceHead, ImageRegressor

from .helpers import CountingBackbone


def _configured_tiny_model(config, output_dim):
    model_config = config.get("model", config)
    if model_config.get("head") == "full_covariance":
        head = FullCovarianceHead(6, output_dim, hidden_dim=10, dropout=0.2)
    else:
        head = DeterministicHead(6, output_dim, hidden_dim=10, dropout=0.2)
    return ImageRegressor(CountingBackbone(image_size=8, feature_dim=6), head)


def test_end_to_end_training_and_inference_dry_run(tiny_hdf5_file, tmp_path: Path, monkeypatch):
    monkeypatch.setattr(train_cli, "build_model", _configured_tiny_model)
    config = {
        "seed": 5,
        "paths": {"output_dir": str(tmp_path), "artifact_name": "dry-run.pt"},
        "data": {
            "normalization_source": "synthetic",
            "datasets": {
                "synthetic": {
                    "hdf5_path": str(tiny_hdf5_file),
                    "group": "train",
                    "randomize": True,
                    "shuffle": True,
                },
                "validation": {
                    "hdf5_path": str(tiny_hdf5_file),
                    "group": "validation",
                    "randomize": False,
                    "shuffle": False,
                },
            },
        },
        "experiment": {
            "name": "fixture",
            "detector": "fixture-detector",
            "wavelengths_m": [1e-10],
            "fixed_parameters": {"rot3": 0.0},
        },
        "preprocessing": {"image_size": 8},
        "model": {"backbone": "fixture", "feature_dim": 6, "head_hidden_dim": 10, "dropout": 0.2},
        "uncertainty": {"estimator": "mc_dropout", "samples": 3, "stochastic_scope": "head"},
        "training": {
            "batch_size": 2,
            "num_workers": 0,
            "pin_memory": False,
            "mixed_precision": {"enabled": True, "dtype": "float16"},
            "stages": [
                {
                    "name": "deterministic",
                    "regime": "deterministic",
                    "dataset": "synthetic",
                    "validation_dataset": "validation",
                    "epochs": 4,
                    "learning_rate": 1e-3,
                    "loss": "huber",
                    "early_stopping": {"patience": 2, "restore_best": True},
                },
                {
                    "name": "probabilistic",
                    "regime": "probabilistic",
                    "dataset": "synthetic",
                    "validation_dataset": "validation",
                    "trainable": "head",
                    "epochs": 4,
                    "learning_rate": 1e-3,
                    "loss": "gaussian_nll",
                },
            ],
        },
    }
    artifact, results = train_cli.train_from_config(config, dry_run=True)
    assert artifact.exists()
    assert len(results) == 2
    state, metadata = read_checkpoint(artifact)
    assert metadata is not None
    assert metadata.training["dry_run"] is True
    assert metadata.training["stages"][0]["early_stopping"]["patience"] == 2
    assert metadata.training["results"][0]["best_epoch"] == 1
    assert metadata.training["mixed_precision"] == {
        "enabled": True,
        "dtype": "float16",
        "active": False,
    }
    assert metadata.training["device_type"] == "cpu"
    assert (tmp_path / "logs").is_dir()
    assert metadata.model_config["head"] == "full_covariance"

    monkeypatch.setattr(calibrate_cli, "build_model", _configured_tiny_model)
    geometry, uncertainty = calibrate_cli.infer_image(
        np.ones((8, 8), dtype=np.float32),
        artifact,
        samples=3,
    )
    assert set(geometry) == {"dist", "poni1", "poni2", "rot1", "rot2", "rot3"}
    assert geometry["rot3"] == 0.0
    assert uncertainty is not None
    assert np.asarray(uncertainty["total_covariance"]).shape == (5, 5)

    resumed_artifact, resumed_results = train_cli.train_from_config(
        config,
        checkpoint=str(artifact),
        start_stage="probabilistic",
        dry_run=True,
    )
    assert resumed_artifact == artifact
    assert [result.name for result in resumed_results] == ["probabilistic"]


def test_legacy_six_output_checkpoint_inference(tiny_hdf5_file, tmp_path: Path, monkeypatch):
    legacy_model = ImageRegressor(
        CountingBackbone(image_size=8, feature_dim=6),
        DeterministicHead(6, 6, hidden_dim=10, dropout=0.2),
    )
    checkpoint = tmp_path / "legacy.pth"
    torch.save(legacy_model.state_dict(), checkpoint)
    config = {
        "data": {"hdf5_path": str(tiny_hdf5_file)},
        "detector": "legacy-detector",
        "wavelength": 1e-10,
        "model": {
            "backbone": "fixture",
            "hidden_dim": 6,
            "head_hidden_dim": 10,
            "image_size": 8,
            "num_outputs": 6,
        },
    }
    monkeypatch.setattr(calibrate_cli, "build_model", _configured_tiny_model)
    geometry, uncertainty = calibrate_cli.infer_image(
        np.ones((8, 8), dtype=np.float32),
        checkpoint,
        legacy_config=config,
    )
    assert uncertainty is None
    assert geometry["rot3"] == 0.0
