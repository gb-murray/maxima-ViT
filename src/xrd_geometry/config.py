"""Configuration loading and conversion to typed domain contracts."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml

from .contracts import ExperimentSpec
from .data import parameter_space_from_hdf5
from .preprocessing import ImagePreprocessor


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise TypeError("Configuration root must be a mapping")
    return config


def build_preprocessor(config: Mapping[str, Any]) -> ImagePreprocessor:
    model = config["model"]
    preprocessing = dict(config.get("preprocessing", {}))
    preprocessing.setdefault("image_size", int(model.get("image_size", 1056)))
    return ImagePreprocessor.from_dict(preprocessing)


def resolve_dataset_configs(config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Resolve per-dataset paths while retaining the legacy single-file schema."""

    data = config["data"]
    shared_path = data.get("hdf5_path")
    configured = data.get("datasets")
    if configured is None:
        if shared_path is None:
            raise ValueError("data must define hdf5_path or named datasets")
        return {
            "train": {
                "hdf5_path": shared_path,
                "group": data.get("train_group", "train"),
                "randomize": data.get("randomize_train"),
                "shuffle": True,
            },
            "validation": {
                "hdf5_path": shared_path,
                "group": data.get("validation_group", data.get("test_group", "test")),
                "randomize": False,
                "shuffle": False,
            },
        }

    resolved = {}
    for name, definition in configured.items():
        values = dict(definition)
        values.setdefault("hdf5_path", shared_path)
        if values["hdf5_path"] is None:
            raise ValueError(f"Dataset {name!r} does not define hdf5_path")
        if "group" not in values:
            raise ValueError(f"Dataset {name!r} does not define an HDF5 group")
        resolved[name] = values
    if not resolved:
        raise ValueError("At least one named dataset is required")
    return resolved


def build_experiment(config: Mapping[str, Any]) -> ExperimentSpec:
    experiment = config.get("experiment", {})
    datasets = resolve_dataset_configs(config)
    normalization_source = config["data"].get("normalization_source", next(iter(datasets)))
    if normalization_source not in datasets:
        raise ValueError(f"Unknown normalization_source {normalization_source!r}")
    parameter_space = parameter_space_from_hdf5(
        datasets[normalization_source]["hdf5_path"],
        rot3=float(experiment.get("fixed_parameters", {}).get("rot3", 0.0)),
    )
    wavelengths = experiment.get("wavelengths_m")
    if wavelengths is None and "wavelength" in config:
        wavelengths = [config["wavelength"]]
    if not wavelengths:
        raise ValueError("experiment.wavelengths_m must describe the fixed source spectrum")
    return ExperimentSpec(
        name=experiment.get("name", "xrd_geometry"),
        detector=experiment.get("detector", config.get("detector", "unknown")),
        wavelengths_m=tuple(float(value) for value in wavelengths),
        parameter_space=parameter_space,
        metadata=experiment.get("metadata", {}),
    )
